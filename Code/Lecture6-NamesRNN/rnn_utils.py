from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def tensorToLetter(tensor):
    return all_letters[tensor.argmax().item()]

def tensorToLine(tensor):
    return ''.join([tensorToLetter(tensor[i]) for i in range(tensor.size()[0])])

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import torch.nn as nn


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, category_tensor, line_tensor, learning_rate=0.005, device='cuda'):
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i].to(device), hidden)

    loss = criterion(output, category_tensor)
        
    # Zero the gradients
    optimizer.zero_grad()
    
    # Backward pass:
    loss.backward()
    
    # Update the weights
    optimizer.step()
    
    return output, loss.item()

import time
import math


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_total_length(d):
    return sum(len(v) for v in d.values())

def val_loss(rnn, val_data, criterion = nn.NLLLoss(), device='cuda'):
    loss = 0
    for category in all_categories:
        for line in val_data[category]:
            category_tensor = torch.tensor(
                [all_categories.index(category)], 
                dtype=torch.long,
                device=device)
            line_tensor = lineToTensor(line)
            hidden = rnn.initHidden()
            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(line_tensor[i], 
                                     hidden)            

            loss += criterion(output, category_tensor)
    return loss.item() / get_total_length(val_data)


def train_test_split(data, test_size=0.2, random_state=None):
    if random_state is not None:
        import random
        random.seed(random_state)
    data = list(data)
    random.shuffle(data)
    split = int(len(data) * (1 - test_size))
    return data[:split], data[split:]



def train_loop(rnn, learning_rate=0.005, 
               n_iters=100000, device='cuda'):

    # Split the data into training and validation
    # Take validation and test sets:

    category_lines_train = {}
    category_lines_val = {}
    category_lines_test = {}
    for category in all_categories:
        lines = category_lines[category]
        lines_train, lines_test = train_test_split(lines, test_size=0.05, random_state=42)
        lines_train, lines_val = train_test_split(lines_train, test_size=0.1, random_state=42)
        category_lines_train[category] = lines_train
        category_lines_val[category] = lines_val
        category_lines_test[category] = lines_test


    n_iters = int(n_iters)
    print_every = 5000
    plot_every = 1000
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    val_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # Convert to device:
        category_tensor = category_tensor.to(device)
        line_tensor = line_tensor.to(device)

        output, loss = train(rnn, category_tensor, line_tensor, 
                             learning_rate=learning_rate,
                             device=device)
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
            v_loss = val_loss(rnn, category_lines_val, device=device)
            val_losses.append(v_loss)

    # Get the final test loss:
    test_loss = val_loss(rnn, category_lines_test, device=device)

    return all_losses, val_losses, test_loss

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def confusion_plotter(rnn):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(rnn, line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(rnn, input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

