from flask import Flask, render_template
from flask_socketio import SocketIO
import queue
import simplejson as json
import subprocess
import threading

import minishogilib


class Engine():
    def __init__(self, command=None, cwd=None, verbose=False, usi_option={}, timelimit={}):
        self.verbose = verbose
        self.command = command
        self.usi_option = usi_option
        self.timelimit = timelimit
        self.socketio = None

        self.process = subprocess.Popen(command.split(
        ), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.message_queue = queue.Queue()
        threading.Thread(target=self._message_reader).start()

    def set_socketio(self, socketio):
        self.socketio = socketio

    def _message_reader(self):
        """Receive message from the engine through standard output and store it.
        # Arguments
            verbose: If true, print message in stdout.
        """
        with self.process.stdout:
            for line in iter(self.process.stdout.readline, b''):
                message = line.decode('utf-8').rstrip('\r\n')
                self.message_queue.put(message)

                if self.verbose:
                    if  self.socketio is not None:
                        self.socketio.emit('message', message)
                    print('<:', message)

    def send_message(self, message):
        """Send message to the engine through standard input.
        # Arguments
            message: message sent to the engine.
            verbose: If true, print message in stdout.
        """
        if self.verbose:
            print('>:', message)

        message = (message + '\n').encode('utf-8')
        self.process.stdin.write(message)
        self.process.stdin.flush()

    def readline(self):
        message = self.message_queue.get()
        return message

    def ask_nextmove(self, position, timelimits, byoyomi):
        sfen_position = 'position sfen ' + position.sfen(True)
        command = 'go {} {} {} {} {} {}'.format(
            self.timelimit['btime'], timelimits[0],
            self.timelimit['wtime'], timelimits[1],
            self.timelimit['byoyomi'], byoyomi)

        self.send_message(sfen_position)
        self.send_message(command)

        while True:
            line = self.readline().split()

            if line[0] == 'bestmove':
                return line[1]

    def usi(self):
        self.send_message('usi')

        while True:
            line = self.readline()

            if line == 'usiok':
                break

    def isready(self):
        for (key, value) in self.usi_option.items():
            command = 'setoption name {} value {}'.format(key, value)
            self.send_message(command)

        self.send_message('isready')

        while True:
            line = self.readline()

            if line == 'readyok':
                break

    def usinewgame(self):
        self.send_message('usinewgame')

    def quit(self):
        self.send_message('quit')


def main():
    app = Flask(__name__, template_folder='./')
    app.debug = True
    socketio = SocketIO(app)

    position = minishogilib.Position()
    position.set_start_position()

    with open('settings.json') as f:
        settings = json.load(f)

    engine = Engine(**settings)
    engine.set_socketio(socketio)
    engine.usi()
    engine.isready()

    @app.route('/')
    def sessions():
        return render_template('index.html')

    @socketio.on('command')
    def command(data):
        data = data.split(' ')
        if data[0] == 'newgame':
            position.set_start_position()
            engine.usinewgame()
        elif data[0] == 'move':
            if len(data) < 2:
                socketio.emit('message', 'You have to specify the next move.')
                return

            move = data[1]
            moves = position.generate_moves()
            moves_sfen = [m.sfen() for m in moves]

            if not move in moves_sfen:
                socketio.emit('message', '{} is not a legal move.'.format(move))
                return

            move = position.sfen_to_move(move)
            position.do_move(move)

            display()

        elif data[0] == 'undo':
            position.undo_move()
            display()

        elif data[0] == 'go':
            timelimits = [5000, 5000]

            next_move = engine.ask_nextmove(position, timelimits, 5000)
            next_move = position.sfen_to_move(next_move)
            position.do_move(next_move)

            display()

        else:
            socketio.emit('message', 'Unknown command {}.'.format(data[0]))


    @socketio.on('display')
    def display():
        data = {
            'svg': position.to_svg()
        }

        socketio.emit('display', data)

    socketio.run(app, host='0.0.0.0', port='8000')


if __name__ == '__main__':
    main()
