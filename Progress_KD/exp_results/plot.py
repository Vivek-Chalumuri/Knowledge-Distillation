import pandas as pd
from matplotlib.ticker import MaxNLocator

if __name__ == "__main__":
    df = pd.read_csv("block_acc.csv")
    df = df.set_index('stage')
    ax = df.plot(kind='line', style='x-')
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig = ax.get_figure()
    fig.savefig("pbt_stage_acc.png", bbox_inches='tight')
    # ======================================================
    df = pd.read_csv("progressive_block_exp.csv")
    df = df[df['epoch'] == 10]
    cols = ['alpha', 'test', 'validation']
    df2 = df[cols]
    df2 = df2.set_index('alpha')
    df2 = df2.sort_index()

    ax = df2.plot(kind='line', style='x-')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Accuracy")
    # ax.set_ylim([0.75, 0.9])
    fig = ax.get_figure()
    fig.savefig("pbt_alpha_acc.png", bbox_inches='tight')

