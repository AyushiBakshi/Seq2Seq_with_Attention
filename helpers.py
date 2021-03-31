from __future__ import unicode_literals, print_function, division
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def makePlot(points):
    print('showPlot')
    plt.figure()
    fig, ax = plt.subplots()
    # this locater puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    figure_1 = plt.gcf()
    plt.draw()
    img_path = '/home/arpit9295_2/Ayushi_Workspace/Seq2Seq_with_Attention/img/point_plot'
    figure_1.savefig(img_path)
    plt.show()


def timeLapsed(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
  
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)


def wordIndexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

