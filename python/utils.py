def recvall(sock, data_size):
    data = b''
    while len(data) < data_size:
        packet = sock.recv(data_size - len(data))
        if not packet:
            break

        data += packet

    return data
