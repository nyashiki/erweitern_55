from flask import Flask, render_template
from flask_socketio import SocketIO

import minishogilib

app = Flask(__name__, template_folder='./')
app.debug = True
socketio = SocketIO(app)

position = minishogilib.Position()
position.set_start_position()

@app.route('/')
def sessions():
    return render_template('index.html')

@socketio.on('command')
def command(data):
    if data == 'newgame':
        pass
    elif data == 'move':
        pass
    elif data == 'undo':
        pass
    else:
        socketio.emit('message', 'Unknown command {}.'.format(data))


@socketio.on('display')
def display():
    data = {
        'svg': position.to_svg()
    }

    socketio.emit('display', data)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port='8000')
