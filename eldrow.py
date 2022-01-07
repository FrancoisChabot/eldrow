from collections import Counter
from queue import PriorityQueue
import os
from types import SimpleNamespace


################################################################################
# Usage: Fill in this section and run the script, it will output all possible
#        solutions in order of (subjective) likelyhood, pausing between batches.

# The solution to the wordle
SOLUTION = "banal"

# The guesses that were made. You can simply copy-paste the contents of Wordle's
# "share" feature directly here.
RAW_GUESSES_TXT = """
⬜⬜🟨⬜⬜
⬜🟩⬜⬜⬜
🟨🟩⬜⬜⬜
⬜🟩🟨⬜⬜
🟩🟩🟩🟩🟩
"""

# If you know certain words, you can specify them here.
# e.g.: KNOWN_WORDS = [(0, "tears")]
KNOWN_WORDS = []

# Number of outputs between pauses.
BATCH_LENGTH = 100

################################################################################
# How this works:

# The solver is built of two primary components

# 1. The simplifier
# The simplifier takes a list of workspaces (one for each guess), and tries to
# identify the smallest subset of possible words each guess can possibly have.
#
# However, a typical wordle solve will often have between 1 and 1000 possible
# words for each guess, even when simplified as much as possible. This means
# there are regularly millions of valid inputs that would have led to a given
# solution layout. This leads us to component #2
#
# 2. The searcher
# The searcher visits the graph of possible answers while priorizing
# "high-likelyhood" words, as defined in the companion "words_scored.txt" file.
# That file was pre-baked by combining
# - The list of acceptable words (extracted from wordle's javascript code)
# - A publickly available word/frequncy dataset.
# - Some manual heuristics (such as penalizing words with repeated letters)


# Loads a file containing accepted words (extracted from wordle's javascript)
# Each word has a score associated with it.
# For the file in the repo, the value was computed based on the
# word frequency with a negative bias for repeated letters
def load_words(fname):
    scores = {}
    with open(fname, "r") as file:
        for line in file.readlines():
            w, c = line.strip().split(" ")
            scores[w] = float(c)

    words = list(scores.keys())

    # Words are always sorted by highest to lowest score everywhere
    words.sort(key=lambda x: -scores[x])
    return words, scores


WORDS_FILE = os.path.join(os.path.dirname(__file__), "words_scored.txt")
WORDS, WORD_SCORES = load_words(WORDS_FILE)


#######################
# Simplifier
#
# in-place reduction of the set of possible words at each stage
#######################

def simplify(data):
    while True:
        done = True

        for ws in data:
            done &= filter_by_words(ws)

        done &= propagate_wrong_letters(data)
        done &= constrain_misplaced(data)

        validate(data)

        if done:
            break


# Synchronizes the letters and words in a workspace
def filter_by_words(ws):
    no_op = True
    while True:
        done = True

        # 1. Remove candidates that cannot match available letters.
        def test(word):
            for i, chr in enumerate(word):
                if chr not in ws.letters[i]:
                    return False

            # Conflicting misplaced letters:
            # e.g.: solution=banal, guess = 🟨🟩⬜🟨⬜, word = ladle
            # should be False because the two misplaced letters cannot both be l
            remainders = ws.remainders.copy()
            for i in ws.guess.misplaced:
                if remainders[word[i]] == 0:
                    return False
                remainders[word[i]] -= 1
            return True

        new_c = list(filter(test, ws.candidates))
        no_op &= len(new_c) == len(ws.candidates)
        ws.candidates = new_c

        # 2. Recompute letter sets
        new_letters = [set(), set(), set(), set(), set()]
        for c in ws.candidates:
            for i, chr in enumerate(c):
                new_letters[i].add(chr)

        for i, letter_set in enumerate(new_letters):
            if len(letter_set) != len(ws.letters[i]):
                done = False
                no_op = False
                ws.letters[i] = letter_set

        if done:
            break

    return no_op


def propagate_wrong_letters(data):
    # AAAAAAAAAAAAAHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!!!!

    # It's not as bad as it looks...
    # All this does is ensure that if a letter is flagged as wrong in a guess,
    # Then it is never used anywhere in any *other* guesses.
    no_op = True
    for ws in data:
        for w in ws.guess.wrong:
            if len(ws.letters[w]) == 1:
                l = next(iter(ws.letters[w]))
                for ws2 in data:
                    if ws is not ws2:
                        for l2 in ws2.letters:
                            if l in l2:
                                no_op = False
                                l2.discard(l)

    return no_op


# Determines the possible values that a misplaced letter could have
def get_possible_misplaced(previous, current, next):
    result = set()

    # 1. letters that are now misplaced
    for m in next.guess.misplaced:
        result.update(next.letters[m])

    # 2. letters that are now correct and wern't previously
    for c in next.guess.correct:
        if c not in current.guess.correct:
            result.update(next.letters[c])

    # If we only shuffled misplaced letters around, then they match in both
    # directions.
    if (
        previous
        and previous.guess.correct == current.guess.correct
        and len(previous.guess.misplaced) == len(current.guess.misplaced)
    ):
        result &= set.union(
            *[previous.letters[m] for m in previous.guess.misplaced]
        )
    return result


def constrain_misplaced(data):
    no_op = True
    for i in range(len(data) - 2, 0, -1):
        curr = data[i]
        filter = get_possible_misplaced(
            data[i - 1] if i > 0 else None, curr, data[i + 1]
        )
        for m in curr.guess.misplaced:
            ls = curr.letters[m]
            filtered = ls.intersection(filter)
            if ls != filtered:
                curr.letters[m] = filtered
                no_op = False
    return no_op


def validate(data):
    for ws in data:
        if len(ws.candidates) == 0:
            raise Exception("impossible")

        for l in ws.letters:
            if len(l) == 0:
                raise Exception("impossible")


#######################
# Dijstra search
#######################
def calculate_score(data):
    total = 0.0
    for ws in data:
        # The candidates are always sorted from highest score down
        # so we only need to look at the first one in the list to know
        # the max value.
        total += WORD_SCORES[ws.candidates[0]]
    return total


class SolutionNode:
    def __init__(self, data, depth):
        self.score = calculate_score(data)
        self.data = data
        self.depth = depth

    def __lt__(self, other):
        return self.score > other.score

    def children(self):
        if self.depth == len(self.data):
            return None

        # Important! We simplify the sub-graph as late as possible.
        simplify(self.data)

        result = []

        for word in self.data[self.depth].candidates:
            c_data = [copy_ws(x) for x in self.data]

            set_word(c_data[self.depth], word)
            result.append(c_data)

        return result

    def path(self):
        return [ws.candidates[0] for ws in self.data]


def solutions(data):
    # In order to understand why the algorithm is set up the way it is, 
    # consider that the primary objective is to output the likeliest 
    # solutions as fast as possible  
    # By settings things up as a dijstra search, we know that the results
    # will be produced from highest to lowest likelyhood, which lets us
    # stream them out as they are found, and lets users early-out 
    # after a few solutions.
    
    todo = PriorityQueue()

    todo.put(SolutionNode(data, 0))

    while not todo.empty():
        next = todo.get()
        try:
            children = next.children()
            if children:
                for c_data in children:
                    todo.put(SolutionNode(c_data, next.depth + 1))
            else:
                yield (next.path(), next.score)
        except:
            # Ideally, simplify() should be catching every single scenarios
            # that leads here...
            pass

#######################
# Solver entrypoint
#######################


def solve(solution, guesses, known_words):
    data = [init_ws(solution, g) for g in guesses]

    for i, w in known_words:
        set_word(data[i], w)

    simplify(data)

    return solutions(data)

#######################
# Misc workspace utilities
#######################


def init_ws(solution, guess):
    alphabet = set("abcdefghijklmnopqrstuvwxyz")
    letters = [None, None, None, None, None]

    remainders = Counter(solution)

    def remove_letter(c):
        remainders[c] -= 1
        if remainders[c] == 0:
            del remainders[c]

    for i in guess.correct:
        letters[i] = set(solution[i])
        remove_letter(solution[i])

    for i in guess.misplaced:
        l = set(remainders.keys())
        l.discard(solution[i])
        letters[i] = l

    wrong_l = alphabet.difference(set(remainders.keys()))
    for i in guess.wrong:
        letters[i] = wrong_l.difference(set(solution[i]))

    result = SimpleNamespace(
        remainders=remainders,
        guess=guess,
        letters=letters,
        candidates=WORDS.copy(),
    )
    return result


# Lighter-weight deep copy for a guess' workspace
def copy_ws(ws):
    result = SimpleNamespace(
        remainders=ws.remainders,
        guess=ws.guess,
        letters=[x.copy() for x in ws.letters],
        candidates=list(ws.candidates),
    )
    return result


def set_word(ws, word):
    ws.candidates = [word]
    for i, chr in enumerate(word):
        ws.letters[i] = set(chr)

#######################
# Input interpretation
#######################


def make_guess(raw_guess):
    result = SimpleNamespace(correct=[], misplaced=[], wrong=[])

    for i, c in enumerate(raw_guess):
        if c == "🟩":
            result.correct.append(i)
        elif c == "🟨":
            result.misplaced.append(i)
        elif c == "⬜":
            result.wrong.append(i)
        else:
            raise Exception("Badly formatted guess")

    return result


if __name__ == "__main__":
    raw_guesses = [s.strip()
                   for s in RAW_GUESSES_TXT.splitlines() if s.strip()]
    guesses = list(map(make_guess, raw_guesses))

    for i, s in enumerate(solve(SOLUTION, guesses, KNOWN_WORDS)):
        print(s)
        if i != 0 and i % BATCH_LENGTH == 0:
            print("...paused, press enter to continue...")
            input()
