from termcolor import colored


def output_error(msg):
    print(colored("[Error] " + str(msg), 'red'))


def output_debug(msg):
    print(colored("[Debug] " + str(msg), 'yellow'))


def output_info(msg):
    print(colored("[Info] " + str(msg), 'green'))
