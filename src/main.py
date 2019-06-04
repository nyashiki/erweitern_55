from ctypes import cdll

def main():
    lib = cdll.LoadLibrary('../minishogi_lib/target/release/libminishogi_lib.so')
    lib.hello_world()

if __name__ == '__main__':
    main()