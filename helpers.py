from __future__ import unicode_literals, print_function, division
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def showPlot(points):
    print('showPlot')
    plt.figure()
    fig, ax = plt.subplots()
    # this locater puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    # return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    return '%s' % (asMinutes(s))


def wordIndexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

