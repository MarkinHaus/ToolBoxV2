import os

from toolboxv2 import AppArgs

NAME = 'docker'


def run(app, args: AppArgs):
    ford_build = ''
    if args.build:
        ford_build = ' --build'
    comm = ''
    if args.modi == 'test':
        comm = 'docker compose up test' + ford_build
    if args.modi == 'live':
        comm = 'docker compose up live' + ford_build
    if args.modi == 'dev':
        comm = 'docker compose up dev --watch' + ford_build
    app.print(f"Running command : {comm}")
    try:
        os.system(comm)
    except KeyboardInterrupt:
        app.print("Exit")

