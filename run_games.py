import re
import os
import subprocess
import sys
import time
import argparse
import features
import shutil
import threading

class GameParams:
    def __init__(self, hash, threads, games_per_round, time_per_move=None, time_increment_per_move=None, nodes_per_move=None):
        self.hash = hash
        self.threads = threads
        self.games_per_round = games_per_round
        self.time_per_move = time_per_move
        self.time_increment_per_move = time_increment_per_move
        self.nodes_per_move = nodes_per_move

        if not time_per_move and not time_increment_per_move and not nodes_per_move:
            raise Exception('Invalid TC specification.')

    def get_all_params(self):
        params = []

        params += ['-each']

        params += [
            f'option.Hash={self.hash}',
            f'option.Threads={self.threads}'
        ]

        if self.nodes_per_move:
            params += [
                f'tc=10000+10000',
                f'nodes={self.nodes_per_move}',
            ]
        else:
            inc = self.time_increment_per_move or 0
            params += [f'tc={self.time_per_move}+{inc}']

        params += ['-games', f'{self.games_per_round}']

        return params


def convert_ckpt(root_dir,features):
    """ Find the list of checkpoints that are available, and convert those that have no matching .nnue """
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt, or epoch=3-step=321151.ckpt
    p = re.compile("epoch.*\.ckpt")
    ckpts = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                ckpts.append(os.path.join(path, filename))

    # lets move the .nnue files a bit up in the tree, and get rid of the = sign.
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt -> run96/run0/nn-epoch3.nnue
    for ckpt in ckpts:
        nnue_file_name = re.sub("default/version_[0-9]+/checkpoints/", "", ckpt)
        nnue_file_name = re.sub(r"epoch\=([0-9]+).*\.ckpt", r"nn-epoch\1.nnue", nnue_file_name)
        if not os.path.exists(nnue_file_name) and os.path.exists(ckpt):
            with subprocess.Popen([sys.executable, 'serialize.py', ckpt, nnue_file_name, f'--features={features}']) as process:
                if process.wait():
                    print("Error serializing!")


def find_nnue(root_dir):
    """ Find the set of nnue nets that are available for testing, going through the full subtree """
    p = re.compile("nn-epoch[0-9]*.nnue")
    nnues = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                nnues.append(os.path.join(path, filename))
    return nnues


def parse_ordo(root_dir, nnues):
    """ Parse an ordo output file for rating and error """
    ordo_file_name = os.path.join(root_dir, "ordo.out")
    ordo_scores = {}
    for name in nnues:
        ordo_scores[name] = (-500, 1000)

    if os.path.exists(ordo_file_name):
        ordo_file = open(ordo_file_name, "r")
        lines = ordo_file.readlines()
        for line in lines:
            if "nn-epoch" in line:
                fields = line.split()
                net = fields[1]
                rating = float(fields[3])
                error = float(fields[4])
                ordo_scores[net] = (rating, error)

    return ordo_scores


def run_match(best, root_dir, c_chess_exe, concurrency, book_file_name, stockfish_base, stockfish_test, game_params):
    """ Run a match using c-chess-cli adding pgns to a file to be analysed with ordo """
    pgn_file_name = os.path.join(root_dir, "out_temp.pgn")
    command = []
    if sys.platform != "win32":
        command += ['stdbuf -o0']
    command += [
        c_chess_exe,
        '-gauntlet', '-rounds', '1',
        '-concurrency', f'{concurrency}'
    ]
    command += game_params.get_all_params()
    command += [
        '-openings', f'file={book_file_name}', 'order=random', '-repeat',
        '-resign', 'count=3', 'score=700',
        '-draw', 'count=8', 'score=10',
        '-pgn', f'{pgn_file_name}', '0'
    ]
    command += ['-engine', f'cmd={stockfish_base}', 'name=master']
    for net in best:
        evalfile = os.path.join(os.getcwd(), net)
        command += ['-engine', f'cmd={stockfish_test}', f'name={net}', f'option.EvalFile={evalfile}']

    print("Running match with c-chess-cli ... {}".format(pgn_file_name), flush=True)
    c_chess_out = open(os.path.join(root_dir, "c_chess.out"), 'w')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    seen = {}
    for line in process.stdout:
        line = line.decode('utf-8')
        c_chess_out.write(line)
        if 'Score' in line:
            epoch_num = re.search(r'epoch(\d+)', line)
            if epoch_num.group(1) not in seen:
                sys.stdout.write('\n')
            seen[epoch_num.group(1)] = True
            sys.stdout.write('\r' + line.rstrip())
            sys.stdout.flush()
    sys.stdout.write('\n')
    c_chess_out.close()
    if process.wait() != 0:
        print("Error running match!")

    print("Finished running match.")

def run_ordo(root_dir, ordo_exe, concurrency):
    """ run an ordo calcuation on an existing pgn file """
    pgn_file_name = os.path.join(root_dir, "out.pgn")
    ordo_file_name = os.path.join(root_dir, "ordo.out")
    ordo_file_name_temp = os.path.join(root_dir, "ordo_temp.out")
    command = [
        ordo_exe,
        '-q',
        '-g',
        '-J',
        '-p', f'{pgn_file_name}',
        '-a', '0.0',
        '--anchor=master',
        '--draw-auto',
        '--white-auto',
        '-s', '100',
        f'--cpus={concurrency}',
        '-o', f'{ordo_file_name_temp}'
    ]

    print("Running ordo ranking ... {}".format(ordo_file_name), flush=True)
    with subprocess.Popen(command) as process:
        if process.wait():
            print("Error running ordo!")
        else:
            os.replace(ordo_file_name_temp, ordo_file_name)

    print("Finished running ordo.")

def run_round(
    root_dir,
    explore_factor,
    ordo_exe,
    c_chess_exe,
    stockfish_base,
    stockfish_test,
    book_file_name,
    concurrency,
    features,
    game_params
):
    """ run a round of games, finding existing nets, analyze an ordo file to pick most suitable ones, run a round, and run ordo """

    # find and convert checkpoints to .nnue
    convert_ckpt(root_dir, features)

    # find a list of networks to test
    nnues = find_nnue(root_dir)
    if len(nnues) == 0:
        print("No .nnue files found in {}".format(root_dir))
        time.sleep(10)
        return
    else:
        print("Found {} nn-epoch*.nnue files".format(len(nnues)))

    # Get info from ordo data if that is around
    ordo_scores = parse_ordo(root_dir, nnues)

    # provide the top 3 nets
    print("Best nets so far:")
    ordo_scores = dict(
        sorted(ordo_scores.items(), key=lambda item: item[1][0], reverse=True)
    )
    count = 0
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0], ordo_scores[net][1]))
        count += 1
        if count == 3:
            break

    # get top 3 with error bar added, for further investigation
    print("Measuring nets:")
    ordo_scores = dict(
        sorted(
            ordo_scores.items(),
            key=lambda item: item[1][0] + explore_factor * item[1][1],
            reverse=True,
        )
    )
    best = []
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0], ordo_scores[net][1]))
        best.append(net)
        if len(best) == 3:
            break

    # run these nets against master...
    # and run a new ordo ranking for the games played so far
    run_match_thread = threading.Thread(
        target=run_match,
        args=(best, root_dir, c_chess_exe, concurrency, book_file_name, stockfish_base, stockfish_test, game_params)
    )
    run_ordo_thread = threading.Thread(
        target=run_ordo,
        args=(root_dir, ordo_exe, concurrency)
    )
    run_match_thread.start()
    run_ordo_thread.start()

    run_match_thread.join()
    run_ordo_thread.join()

    # we write current match info to a temporary file and then copy back
    # to allow ordo to run in parallel (since it's single threaded and may take a long time)
    # we then append the new games to the main file
    main_pgn_file_name = os.path.join(root_dir, "out.pgn")
    curr_pgn_file_name = os.path.join(root_dir, "out_temp.pgn")
    if not os.path.exists(main_pgn_file_name):
        with open(main_pgn_file_name, 'w'): pass
    try:
        with open(main_pgn_file_name, 'a') as file_to:
            with open(curr_pgn_file_name, 'r') as file_from:
                for line in file_from:
                    file_to.write(line)
    except:
        print('Something went wrong when adding new games to the main file.')

def main():
    # basic setup
    parser = argparse.ArgumentParser(
        description="Finds the strongest .nnue / .ckpt in tree, playing games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="""The directory where to look, recursively, for .nnue or .ckpt.
                 This directory will be used to store additional files,
                 in particular the ranking (ordo.out)
                 and game results (out.pgn and c_chess.out).""",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrently running threads",
    )
    parser.add_argument(
        "--explore_factor",
        default=1.5,
        type=float,
        help="`expected_improvement = rating + explore_factor * error` is used to pick select the next networks to run.",
    )
    parser.add_argument(
        "--ordo_exe",
        type=str,
        default="./ordo",
        help="Path to ordo, see https://github.com/michiguel/Ordo",
    )
    parser.add_argument(
        "--c_chess_exe",
        type=str,
        default="./c-chess-cli",
        help="Path to c-chess-cli, see https://github.com/lucasart/c-chess-cli",
    )
    parser.add_argument(
        "--stockfish_base",
        type=str,
        default="./stockfish",
        help="Path to stockfish master (reference version), see https://github.com/official-stockfish/Stockfish",
    )
    parser.add_argument(
        "--stockfish_test",
        type=str,
        help="(optional) Path to new stockfish binary, if not set, will use stockfish_base",
    )
    parser.add_argument(
        "--book_file_name",
        type=str,
        default="./noob_3moves.epd",
        help="Path to a suitable book, see https://github.com/official-stockfish/books",
    )
    parser.add_argument(
        "--time_per_move",
        type=float,
        default=4.0
    )
    parser.add_argument(
        "--time_increment_per_move",
        type=float,
        default=0.04
    )
    parser.add_argument(
        "--nodes_per_move",
        type=int,
        default=None,
        help="Overrides time per move."
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=8
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1
    )
    parser.add_argument(
        "--games_per_round",
        type=int,
        default=200
    )
    features.add_argparse_args(parser)
    args = parser.parse_args()

    stockfish_base = args.stockfish_base
    stockfish_test = args.stockfish_test
    if stockfish_test is None:
        stockfish_test = stockfish_base

    if not shutil.which(stockfish_base):
       sys.exit("Stockfish base is not executable !")

    if not shutil.which(stockfish_test):
       sys.exit("Stockfish test is not executable!")

    if not shutil.which(args.ordo_exe):
       sys.exit("ordo is not executable!")

    if not shutil.which(args.c_chess_exe):
       sys.exit("c_chess_cli is not executable!")

    if not os.path.exists(args.book_file_name):
       sys.exit("book does not exist!")

    while True:
        run_round(
            args.root_dir,
            args.explore_factor,
            args.ordo_exe,
            args.c_chess_exe,
            stockfish_base,
            stockfish_test,
            args.book_file_name,
            args.concurrency,
            args.features,
            GameParams(args.hash, args.threads, args.games_per_round, args.time_per_move, args.time_increment_per_move, args.nodes_per_move)
        )


if __name__ == "__main__":
    main()
