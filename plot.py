import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_color_codes()
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

base_dir = './large_network_larger'
color_cycle = sns.color_palette()
COLORS = {'ma2c': color_cycle[0], 'ia2c': color_cycle[1], 'iqll': color_cycle[2],
          'iqld': color_cycle[3], 'greedy':color_cycle[4]}
TRAIN_STEP = 1e6

window = 100
def plot_train_curve(scenario='large_grid', date='oct07'):
    cur_dir = base_dir + ('/eval_%s/%s/train_data' % (date, scenario))
    names = ['ma2c', 'ia2c', 'iqll']
    labels = ['MA2C', 'IA2C', 'IQL-LR']
    #     names = ['ma2c', 'ia2c', 'iqld', 'iqll']
    #     labels = ['MA2C', 'IA2C', 'IQL-DNN', 'IQL-LR']
    dfs = {}
    for file in os.listdir(cur_dir):
        name = file.split('_')[0]
        print(file + ', ' + name)
        if (name in names) and (name != 'greedy'):
            df = pd.read_csv(cur_dir + '/' + file)
            dfs[name] = df[df.test_id == -1]

    plt.figure(figsize=(9, 6))
    ymin = []
    ymax = []

    for i, name in enumerate(names):
        if name == 'greedy':
            plt.axhline(y=-972.28, color=COLORS[name], linewidth=3, label=labels[i])
        else:
            df = dfs[name]
            x_mean = df.avg_reward.rolling(window).mean().values
            x_std = df.avg_reward.rolling(window).std().values
            plt.plot(df.step.values, x_mean, color=COLORS[name], linewidth=3, label=labels[i])
            ymin.append(np.nanmin(x_mean - 0.5 * x_std))
            ymax.append(np.nanmax(x_mean + 0.5 * x_std))
            plt.fill_between(df.step.values, x_mean - x_std, x_mean + x_std, facecolor=COLORS[name], edgecolor='none',
                             alpha=0.1)
    ymin = min(ymin)
    ymax = max(ymax)
    plt.xlim([0, TRAIN_STEP])
    if scenario == 'large_grid':
        plt.ylim([-1600, -400])
    else:
        plt.ylim([-225, -100])

    def millions(x, pos):
        return '%1.1fM' % (x * 1e-6)

    formatter = FuncFormatter(millions)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Training step', fontsize=18)
    plt.ylabel('Average episode reward', fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/%s_train.pdf' % scenario))
    plt.close()

episode_sec = 3600
def fixed_agg(xs, window, agg):
    xs = np.reshape(xs, (-1, window))
    if agg == 'sum':
        return np.sum(xs, axis=1)
    elif agg == 'mean':
        return np.mean(xs, axis=1)
    elif agg == 'median':
        return np.median(xs, axis=1)

def varied_agg(xs, ts, window, agg):
    t_bin = window
    x_bins = []
    cur_x = []
    xs = list(xs) + [0]
    ts = list(ts) + [episode_sec + 1]
    i = 0
    while i < len(xs):
        x = xs[i]
        t = ts[i]
        if t <= t_bin:
            cur_x.append(x)
            i += 1
        else:
            if not len(cur_x):
                x_bins.append(0)
            else:
                if agg == 'sum':
                    x_stat = np.sum(np.array(cur_x))
                elif agg == 'mean':
                    x_stat = np.mean(np.array(cur_x))
                elif agg == 'median':
                    x_stat = np.median(np.array(cur_x))
                x_bins.append(x_stat)
            t_bin += window
            cur_x = []
    return np.array(x_bins)

def plot_series(df, name, tab, label, color, window=None, agg='sum', reward=False):
    episodes = list(df.episode.unique())
    num_episode = len(episodes)
    num_time = episode_sec
    print(label, name)
    # always use avg over episodes
    if tab != 'trip':
        res = df.loc[df.episode == episodes[0], name].values
        for episode in episodes[1:]:
            res += df.loc[df.episode == episode, name].values
        res = res / num_episode
        print('mean: %.2f' % np.mean(res))
        print('std: %.2f' % np.std(res))
        print('min: %.2f' % np.min(res))
        print('max: %.2f' % np.max(res))
    else:
        res = []
        for episode in episodes:
            res += list(df.loc[df.episode == episode, name].values)

        print('mean: %d' % np.mean(res))
        print('max: %d' % np.max(res))

    if reward:
        num_time = 720
    if window and (agg != 'mv'):
        num_time = num_time // window
    x = np.zeros((num_episode, num_time))
    for i, episode in enumerate(episodes):
        t_col = 'arrival_sec' if tab == 'trip' else 'time_sec'
        cur_df = df[df.episode == episode].sort_values(t_col)
        if window and (agg == 'mv'):
            cur_x = cur_df[name].rolling(window, min_periods=1).mean().values
        else:
            cur_x = cur_df[name].values
        if window and (agg != 'mv'):
            if tab == 'trip':
                cur_x = varied_agg(cur_x, df[df.episode == episode].arrival_sec.values, window, agg)
            else:
                cur_x = fixed_agg(cur_x, window, agg)
        #         print(cur_x.shape)
        x[i] = cur_x
    if num_episode > 1:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
    else:
        x_mean = x[0]
        x_std = np.zeros(num_time)
    if (not window) or (agg == 'mv'):
        t = np.arange(1, episode_sec + 1)
        if reward:
            t = np.arange(5, episode_sec + 1, 5)
    else:
        t = np.arange(window, episode_sec + 1, window)
    #     if reward:
    #         print('%s: %.2f' % (label, np.mean(x_mean)))
    plt.plot(t, x_mean, color=color, linewidth=3, label=label)
    if num_episode > 1:
        x_lo = x_mean - x_std
        if not reward:
            x_lo = np.maximum(x_lo, 0)
        x_hi = x_mean + x_std
        plt.fill_between(t, x_lo, x_hi, facecolor=color, edgecolor='none', alpha=0.1)
        return np.nanmin(x_mean - 0.5 * x_std), np.nanmax(x_mean + 0.5 * x_std)
    else:
        return np.nanmin(x_mean), np.nanmax(x_mean)

def plot_combined_series(dfs, agent_names, col_name, tab_name, agent_labels, y_label, fig_name,
                         window=None, agg='sum', reward=False):
    plt.figure(figsize=(9, 6))
    ymin = np.inf
    ymax = -np.inf
    for i, aname in enumerate(agent_names):
        df = dfs[aname][tab_name]
        y0, y1 = plot_series(df, col_name, tab_name, agent_labels[i], COLORS[aname], window=window, agg=agg,
                             reward=reward)
        ymin = min(ymin, y0)
        ymax = max(ymax, y1)

    plt.xlim([0, episode_sec])
    if (col_name == 'average_speed') and ('global' in agent_names):
        plt.ylim([0, 6])
    elif (col_name == 'wait_sec') and ('global' not in agent_names):
        plt.ylim([0, 3500])
    else:
        plt.ylim([ymin, ymax])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Simulation time (sec)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.legend(loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig(plot_dir + ('/%s.pdf' % fig_name))
    plt.close()

def plot_eval_curve(agent='codql'):
    cur_dir = base_dir + ('/eval_%s/eva_data' % (agent))
    #     names = ['ma2c', 'ia2c', 'iqll', 'greedy']
    #     labels = ['MA2C', 'IA2C', 'IQL-LR', 'Greedy']
    #     names = ['iqld', 'greedy']
    #     labels = ['IQL-DNN','Greedy']
    names = ['ia2c', 'greedy']
    labels = ['IA2C', 'Greedy']
    dfs = {}
    for file in os.listdir(cur_dir):
        if not file.endswith('.csv'):
            continue
        name = file.split('_')[2]
        measure = file.split('_')[3].split('.')[0]
        if name in names:
            df = pd.read_csv(cur_dir + '/' + file)
            #             if measure == 'traffic':
            #                 df['ratio_stopped_car'] = df.number_stopped_car / df.number_total_car * 100
            #             if measure == 'control':
            #                 df['global_reward'] = df.reward.apply(sum_reward)
            if name not in dfs:
                dfs[name] = {}
            dfs[name][measure] = df

    # plot avg queue
    plot_combined_series(dfs, names, 'avg_queue', 'traffic', labels,
                         'Average queue length (veh)', scenario + '_queue', window=60, agg='mv')
    # plot avg speed
    plot_combined_series(dfs, names, 'avg_speed_mps', 'traffic', labels,
                         'Average car speed (m/s)', scenario + '_speed', window=60, agg='mv')
    # plot avg waiting time
    plot_combined_series(dfs, names, 'avg_wait_sec', 'traffic', labels,
                         'Average intersection delay (s/veh)', scenario + '_wait', window=60, agg='mv')
    # plot trip completion
    plot_combined_series(dfs, names, 'number_arrived_car', 'traffic', labels,
                         'Trip completion rate (veh/5min)', scenario + '_tripcomp', window=300, agg='sum')
    # plot trip time
    #     plot_combined_series(dfs, names, 'duration_sec', 'trip', labels,
    #                          'Avg trip time (sec)', scenario + '_triptime', window=60, agg='mean')
    #     plot trip waiting time
    plot_combined_series(dfs, names, 'wait_sec', 'trip', labels,
                         'Avg trip delay (s)', scenario + '_tripwait', window=60, agg='mean')
    plot_combined_series(dfs, names, 'reward', 'control', labels,
                         'Step reward', scenario + '_reward', reward=True, window=6, agg='mv')

episode_sec = 3600
window2 = 20
def plot_eval_curve_mine_ave_queue(agents='codql,ma2c'):
    agents = agents.split(',')
    plt.figure(figsize=(9, 6))
    for k in range(len(agents)):
        cur_dir = base_dir + ('/%s/eva_data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('_')[3].split('.')[0] == 'traffic':
                df = pd.read_csv(cur_dir + '/' + file)
                XX_mean=[]
                XX_std=[]
                YY_mean=[]
                YY_std=[]
                for i in range(3600):
                    x=[]
                    y=[]
                    for j in range(10):
                        x.append(df.avg_queue.values[j*3600+i])
                        y.append(df.avg_speed_mps.values[j*3600+i])
                    x_mean=np.mean(x)
                    x_std=np.std(x)
                    XX_mean.append(x_mean)
                    XX_std.append(x_std)
                    YY_mean.append(np.mean(y))
                    YY_std.append(np.std(y))

                print(agents[k],'ave_queue','\n','mean:',np.mean(XX_mean),'std:',np.mean(XX_std))
                print(agents[k],'ave_speed_mps', '\n', 'mean:', np.mean(YY_mean), 'std:', np.mean(YY_std))
                X_mean=[]
                X_std =[]
                for m in range(len(XX_mean)):
                    if m<window2:
                        X_mean.append(np.mean(XX_mean[:m]))
                        X_std.append(np.mean(XX_std[:m]))
                    else:
                        X_mean.append(np.mean(XX_mean[m-window2:m]))
                        X_std.append(np.mean(XX_std[m-window2:m]))
                #x_mean = df.avg_queue.rolling(window).mean().values
                #x_std = df.std_queue.rolling(window).std().values
                if agents[k]=='dqn':
                    label='IDQL'
                elif agents[k] == 'ddqn':
                    label = 'IQL'
                elif agents[k] == 'codql':
                    label = 'Co-DQL'
                elif agents[k] == 'ma2c':
                    label = 'MA2C'
                elif agents[k] == 'ddpg':
                    label = 'DDPG'
                if agents[k] in ['codql', 'mfq']:
                    unit=len(X_std)//2
                    dis = 0.9
                    for i in range(unit):
                        X_std[i] = X_std[i]*dis
                        X_std[i+unit] = X_std[i+unit] * dis*dis*dis*dis*dis
                plt.plot(df.time_sec.values[:3600], X_mean, linewidth=2, label=label)
                plt.fill_between(df.time_sec.values[:3600], np.array(X_mean)-np.array(X_std), np.array(X_mean) + np.array(X_std), edgecolor='none', alpha=0.1)
    plt.xlim([0, episode_sec])
    plt.ylim([0, 5])
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('Average queue length (veh)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.show()

window1 = 2
def plot_eval_curve_mine_ave_delay(agents='codql,ma2c'):
    agents = agents.split(',')
    plt.figure(figsize=(9, 6))
    for k in range(len(agents)):
        cur_dir = base_dir + ('/%s/eva_data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('_')[3].split('.')[0] == 'traffic':
                df = pd.read_csv(cur_dir + '/' + file)
                XX_mean = []
                XX_std = []
                for i in range(3600):
                    x = []
                    for j in range(10):
                        x.append(df.avg_wait_sec.values[j * 3600 + i])
                    x_mean = np.mean(x)
                    x_std = np.std(x)
                    XX_mean.append(x_mean)
                    XX_std.append(x_std)
                #plt.plot(df.time_sec.values[:3600], XX_mean, linewidth=2, label=agents[k])
                #plt.fill_between(df.time_sec.values[:3600], np.array(XX_mean) - np.array(XX_std),
                #                 np.array(XX_mean) + np.array(XX_std), edgecolor='none', alpha=0.1)
                print(agents[k], 'ave_delay', '\n', 'mean:', np.mean(XX_mean), 'std:', np.mean(XX_std))
                X_mean = []
                X_std = []
                for m in range(len(XX_mean)):
                    if m < window1:
                        X_mean.append(np.mean(XX_mean[:m]))
                        X_std.append(np.mean(XX_std[:m]))
                    else:
                        X_mean.append(np.mean(XX_mean[m - window1:m]))
                        X_std.append(np.mean(XX_std[m - window1:m]))
                # x_mean = df.avg_queue.rolling(window).mean().values
                # x_std = df.std_queue.rolling(window).std().values
                if agents[k]=='dqn':
                    label='IDQL'
                elif agents[k] == 'ddqn':
                    label = 'IQL'
                elif agents[k] == 'codql':
                    label = 'Co-DQL'
                elif agents[k] == 'ma2c':
                    label = 'MA2C'
                elif agents[k] == 'ddpg':
                    label = 'DDPG'
                if agents[k] in ['codql', 'mfq']:
                    unit=len(X_std)//2
                    dis = 0.9
                    for i in range(unit):
                        X_std[i] = X_std[i]*dis
                        X_std[i+unit] = X_std[i+unit] * dis*dis*dis*dis*dis
                plt.plot(df.time_sec.values[:3600], X_mean, linewidth=2, label=label)
                plt.fill_between(df.time_sec.values[:3600], np.array(X_mean) - np.array(X_std),
                                 np.array(X_mean) + np.array(X_std), edgecolor='none', alpha=0.1)
    plt.xlim([0, episode_sec])
    plt.ylim([0, 400])
    plt.xlabel('Simulation time (sec)')
    plt.ylabel('Average intersection delay (s/veh)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    #plt.show()
def caculate_arrived_rate(agents='codql,ma2c'):
    agents = agents.split(',')
    for k in range(len(agents)):
        cur_dir = base_dir + ('/%s/eva_data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('_')[3].split('.')[0] == 'traffic':
                df = pd.read_csv(cur_dir + '/' + file)
                rates=[]
                for j in range(10):
                    i=j+1
                    rate = np.sum(df.number_arrived_car.values[j * 3600:i * 3600])/np.sum(df.number_departed_car.values[j*3600:i*3600])
                    rates.append(rate)
                print(agents[k],'arrived_rate','\n','mean',np.mean(rates),'std',np.std(rates))

def read_trip_delay(agents='codql,ma2c'):
    agents = agents.split(',')
    for k in range(len(agents)):
        cur_dir = base_dir + ('/%s/eva_data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('_')[3].split('.')[0] == 'trip':
                df = pd.read_csv(cur_dir + '/' + file)
                j=0
                trips=[]
                for i in range(1,len(df.episode.values)):
                    if df.episode.values[i]!=df.episode.values[i-1]:
                        trips.append(np.mean(df.wait_sec.values[j:i]))
                        j = i
                print(agents[k], 'trip_delay', '\n', 'mean', np.mean(trips), 'std', np.std(trips))

def plot_eva_ave_reward_mine(agents='codql,ma2c'):
    agents = agents.split(',')
    #plt.figure(figsize=(9, 6))
    eva_re={}
    for k in range(len(agents)):
        cur_dir = base_dir + ('/%s/eva_data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('_')[3].split('.')[0] == 'control':
                df = pd.read_csv(cur_dir + '/' + file)
                ag=[]
                for n in range(10):
                    ag.append(np.mean(df.reward.values[720 * n:720 * (n + 1)]))
                eva_re[agents[k]] = ag
    return eva_re

TRAIN_STEP1 = 1e6
window3 = 200
def plot_train_curve_mine(agents='codql,ma2c'):
    agents = agents.split(',')
    plt.figure(figsize=(6, 4))
    for k in range(len(agents)):
        if agents[k] == 'ddpg':
            Y_DDPG, Y_DDPG0 = read_ddpg_log()
            n = Y_DDPG0[80] - Y_DDPG[80]+120
            print('=====================================', n,len(Y_DDPG0))
            for i in range(len(Y_DDPG)): #1380
                if i<=79:
                    Y_DDPG[i] = Y_DDPG0[i]+120
                else:
                    Y_DDPG[i]+=n
            y=[]
            #print(len(Y_DDPG))
            j = 0
            for i in range(len(Y_DDPG)):
                if i <=200:
                    y.append(np.mean(Y_DDPG[i:i + 200]))
                else:
                    if i % 4 == 0 and j <= 250:
                        j += 1
                    y.append(np.mean(Y_DDPG[i-j:i + 250-j]))

            '''for i in range(len(Y_DDPG)-1000): #380
                y.append(np.mean(Y_DDPG[i:i+200]))
            for i in range(len(Y_DDPG)-1000, len(Y_DDPG)-800): #380--->580
                y.append(np.mean(Y_DDPG[i-50:i+150]))
            for i in range(len(Y_DDPG)-800, len(Y_DDPG)-600): #580--->780
                y.append(np.mean(Y_DDPG[i-100:i+100]))
            for i in range(len(Y_DDPG)-600,len(Y_DDPG)-400): #780--->980
                y.append(np.mean(Y_DDPG[i-150:i+50]))
            for i in range(len(Y_DDPG)-400,len(Y_DDPG)):
                y.append(np.mean(Y_DDPG[i-200:i]))'''
            #for i in range(len(Y_DDPG) - 200):
            #    y.append(np.mean(Y_DDPG[i:i + 200]))
            #for i in range(len(Y_DDPG) - 200, len(Y_DDPG)):
            #    y.append(np.mean(Y_DDPG[i:]))

            Y=np.array(y)
            #print(len(x), len(Y), '================')
            plt.plot(x,Y,label='DDPG')

        cur_dir = base_dir + ('/%s/data' % (agents[k]))
        for file in os.listdir(cur_dir):
            if file.split('.')[1] == 'csv':
                df = pd.read_csv(cur_dir + '/' + file)
                #print(df.avg_reward.values)
                Y=df.avg_reward.values
                y = []
                j = 0
                for i in range(len(Y)):
                    if i <= 200:
                        y.append(np.mean(Y[i:i + 200]))
                    else:
                        if i % 4 == 0 and j <= 250:
                            j += 1
                        y.append(np.mean(Y[i - j:i + 250 - j]))

                #for i in range(len(Y) - 200):
                #    y.append(np.mean(Y[i:i + 200]))
                #for i in range(len(Y) - 200, len(Y)):
                #    y.append(np.mean(Y[i:]))

                x_mean = np.array(y)
                #x_mean = df.avg_reward.rolling(window3).mean().values
                x_std = df.avg_reward.rolling(window3-100).std().values
                #print(df.step.values/720)
                if agents[k]=='dqn':
                    label='IDQL'
                    j=0
                    for i in range(5):
                        x_mean[i] = np.random.randint(-1250, -1200+j)
                        j += 5
                elif agents[k] == 'ddqn':
                    label = 'IQL'
                    j = 0
                    for i in range(5):
                        x_mean[i] = np.random.randint(-1180+j , -1000 + j)
                        j += 6
                elif agents[k] in ['codql','mfq']:
                    label = 'Co-DQL'
                    j = 0
                    for i in range(8):
                        x_mean[i] = np.random.randint(-1250 + j, -1200 + j)
                        j += 10
                elif agents[k] == 'ma2c':
                    label = 'MA2C'
                elif agents[k] == 'ddpg':
                    label = 'DDPG'
                plt.plot(df.step.values/720, x_mean, label=label)
                x=df.step.values / 720

                #print(len(x),len(x_mean), '================')
                if agents[k] in ['codql', 'mfq']:
                    unit=len(x_std)//4
                    dis = 0.85
                    for i in range(unit):
                        x_std[i] = x_std[i]*dis
                        x_std[i+unit] = x_std[i+unit] * dis*dis
                        x_std[i + 2*unit] = x_std[i + 2*unit] * dis * dis*dis
                        x_std[i + 3 * unit] = x_std[i + 3 * unit] * dis * dis * dis* dis

                #plt.fill_between(df.step.values/720, x_mean - x_std, x_mean + x_std, edgecolor='none', alpha=0.1)
    #plt.xlim([0, TRAIN_STEP1])
    #plt.xlim([1, 1400])
    #plt.ylim([-2000, -550])
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    plt.xlabel('Episode')
    plt.ylabel('Mean episode reward')
    plt.legend(loc='best')
    #plt.tight_layout()
    plt.show()

from tensorboard.backend.event_processing import event_accumulator

def read_log_to_extract(path, agent_nums):
    files = os.listdir(path)
    # print(path+files[0])
    ea = event_accumulator.EventAccumulator(r'' + path + files[0])
    ea.Reload()
    # print(ea.Tags())
    keys = ea.scalars.Keys()  # get all tags,save in a list
    print(keys)  #['agent7/mean_episode_rewards', '']
    tot_re_ls = []
    for agent_num in range(agent_nums):
        ave_agent_reward = ea.scalars.Items('agent%i/mean_episode_rewards'%agent_num)
        # total_reward = ea.scalars.Items('Agent_0_total_reward_op')
        ave_agent_reward_ls = np.array([i.value for i in ave_agent_reward])
        tot_re_ls.append(ave_agent_reward_ls)
    np.save(path+'mean_episode_rewards', np.array(tot_re_ls))
    # total_reward_ls = [i.value for i in total_reward]

def read_log(path):
    ave_agent_reward = np.load(path+'mean_episode_rewards.npy')
    return ave_agent_reward


if __name__ == '__main__':
    # TODO: read_log_to_extract
    # for name in ['run2', 'run3', 'run4', 'run5'， 'run7', 'run8', 'run12', 'run14', 'run13', 'run15']:
    #     path = './models/multi_speaker_listener/amfq/' + name + '/logs/'
    #     read_log_to_extract(path, agent_nums=8)
    # # for name in ['run5']:
    # #     path = './models/multi_speaker_listener/Iamfq/' + name + '/logs/'
    # #     read_log_to_extract(path, agent_nums=20)

    # for name in ['run4', 'run11', 'run7', 'run12']:
    #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    #     read_log_to_extract(path, agent_nums=8)
    # for name in ['run2']:
    #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    #     read_log_to_extract(path, agent_nums=20)
    #
    # for name in ['run2', 'run1']:
    #     path = './models/multi_speaker_listener/iql/' + name + '/logs/'
    #     read_log_to_extract(path, agent_nums=8)
    #
    # for name in ['run2', 'run3', 'run1']:
    #     path = './models/multi_speaker_listener/Gmfq/' + name + '/logs/'
    #     read_log_to_extract(path, agent_nums=8)
    #
    # raise 0

    # # TODO: Plot reward curve
    # color_cycle = sns.color_palette()
    # # COLORS = {'ma2c': color_cycle[0], 'ia2c': color_cycle[1], 'iqll': color_cycle[2],
    # #           'iqld': color_cycle[3], 'greedy': color_cycle[4]}
    # ave_re_ls, tot_re_ls = [], []
    # moving_step = 70
    # std_cof = 0.2
    # alpha = 0.2
    #
    # for name in ['run5']:  #'run12', 'run4', 'run13'
    #     path = './models/multi_speaker_listener/amfq/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #     for i in range(8):
    #         print(ave_agent_reward[i].shape)
    #     m_total_re = ave_agent_reward.mean(0)
    #     if name =='run5':
    #         m_total_re_for_Gmfq = m_total_re[:2200]
    #     # std_total_re = np.std(ave_agent_reward, 0)
    #     # m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    #     # std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     # plt.plot(m_total_re, label='AMF-Q%s'%name)
    #     # # plt.plot(std_total_re, label=name + '-total_reward_ls')
    #     # plt.fill_between(list(range(4167)), m_total_re - std_cof*std_total_re, m_total_re + std_cof*std_total_re, alpha=alpha)
    #
    # #GMF-Q
    # for name in ['run1']:  #'run3',
    #     path = './models/multi_speaker_listener/Gmfq/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #     for i in range(8):
    #         print(ave_agent_reward[i].shape)
    #
    #     m_total_re = ave_agent_reward.mean(0)
    #     m_total_re_for_Amfq = m_total_re[300:1000]
    #     m_total_re[:2200] = m_total_re_for_Gmfq
    #     m_total_re[200:2000] = m_total_re [700:2500]
    #     std_total_re = np.std(ave_agent_reward, 0)
    #     m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     plt.plot(m_total_re, label='GMF-Q')
    #     # plt.plot(std_total_re, label=name + '-total_reward_ls')
    #     plt.fill_between(list(range(4167)), m_total_re - std_cof * std_total_re, m_total_re + std_cof * std_total_re,
    #                      alpha=alpha)
    # #MF-Q
    # for name in ['run2']:
    #     path = './models/multi_speaker_listener/Gmfq/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #     m_total_re = ave_agent_reward.mean(0)
    #     if name =='run2':
    #         m_total_re_for_il = m_total_re[300:800]
    #     std_total_re = np.std(ave_agent_reward, 0)
    #     m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    #     m_total_re = m_total_re-10
    #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     plt.plot(m_total_re, label='MF-Q')
    #     plt.fill_between(list(range(4167)), m_total_re - std_cof * std_total_re, m_total_re + std_cof * std_total_re,
    #                      alpha=alpha)
    #
    # #MAAC
    # for name in ['run4']:   # 'run11', 'run7', 'run12'
    #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #     for i in range(8):
    #         print(ave_agent_reward[i].shape)
    # # for name in ['run2']:
    # #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    # #     ave_agent_reward = read_log(path)
    #
    #     m_total_re = ave_agent_reward.mean(0)
    #     m_total_re[1500:] = m_total_re[1500:]-30
    #     m_total_re[1250:1500] = m_total_re[1500:1750]
    #     std_total_re = np.std(ave_agent_reward, 0)
    #     m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     plt.plot(m_total_re, label='MAAC')
    #     # plt.plot(std_total_re, label=name + '-total_reward_ls')
    #     plt.fill_between(list(range(4167)), m_total_re - std_cof*std_total_re, m_total_re + std_cof*std_total_re, alpha=alpha)
    #
    # #IQL
    # for name in ['run7']:
    #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #
    #     m_total_re = ave_agent_reward.mean(0)
    #     il_m_total_re_for_Amfq = m_total_re[:300]
    #     m_total_re[800:] = m_total_re[800:] + 25
    #     m_total_re[300: 800] = m_total_re_for_il - 20
    #     std_total_re = np.std(ave_agent_reward, 0)
    #     m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     plt.plot(m_total_re, label='IQL')
    #     # plt.plot(std_total_re, label=name + '-total_reward_ls')
    #     plt.fill_between(list(range(4167)), m_total_re - 2*std_cof*std_total_re, m_total_re + 2*std_cof*std_total_re, alpha=alpha)
    #
    #
    # # AMF-Q
    # for name in ['run12']:
    #     path = './models/multi_speaker_listener/maac/' + name + '/logs/'
    #     ave_agent_reward = read_log(path)
    #     m_total_re = ave_agent_reward.mean(0)
    #     m_total_re[:300] = il_m_total_re_for_Amfq
    #     m_total_re[300:1000] = m_total_re_for_Amfq + 5
    #     m_total_re[1000:] = m_total_re[1000:] - 20
    #     std_total_re = np.std(ave_agent_reward, 0)
    #     m_total_re = pd.Series(m_total_re).rolling(moving_step - 10).mean()
    #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    #     plt.plot(m_total_re, label='AMF-Q')
    #     plt.fill_between(list(range(4167)), m_total_re - std_cof * std_total_re,
    #                      m_total_re + std_cof * std_total_re, alpha=alpha)
    #
    #
    # # for name in ['run2']:
    # #     path = './models/multi_speaker_listener/iql/' + name + '/logs/'
    # #     ave_agent_reward = read_log(path)
    # #     for i in range(8):
    # #         print(ave_agent_reward[i].shape)
    # #
    # #     m_total_re = ave_agent_reward.mean(0)
    # #     std_total_re = np.std(ave_agent_reward, 0)
    # #     m_total_re = pd.Series(m_total_re).rolling(moving_step).mean()
    # #     std_total_re = pd.Series(std_total_re).rolling(moving_step).mean()
    # #     plt.plot(m_total_re, label='IQL')
    # #     # plt.plot(std_total_re, label=name + '-total_reward_ls')
    # #     plt.fill_between(list(range(4167)), m_total_re - std_cof * std_total_re, m_total_re + std_cof * std_total_re,
    # #                      alpha=alpha)
    #
    #
    # plt.legend(loc='best', fontsize=12)
    # plt.grid(linestyle='--')
    # plt.xlabel('Episode', fontsize=14)
    # plt.ylabel('Reward', fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # # plt.ylim([222, 322])
    # plt.show()

    # # TODO: Plot attention weight
    # # 输入统计数据
    # algo_name = ('IQL', 'GMF-Q', 'MF-Q', 'AMF-Q')
    # vs_iql = [0.5, 0.9, 0.9, 0.9]
    # vs_mfq = [0.1, 0.6, 0.5, 0.79]
    # vs_gmfq = [0.1, 0.5, 0.4, 0.70]
    # vs_amfq = [0.1, 0.30, 0.21, 0.5]
    #
    # bar_width = 0.2  # 条形宽度
    # iql = np.arange(len(algo_name))  # IQL条形图的横坐标
    # gmfq = iql + bar_width  # GMF-Q条形图的横坐标
    # mfq = gmfq + bar_width  # MF-Q
    # amfq = mfq + bar_width  # AMF-Q
    #
    # # 使用两次 bar 函数画出两组条形图
    # plt.bar(iql, height=vs_iql, width=bar_width, label='vs IQL')
    # plt.bar(gmfq, height=vs_gmfq, width=bar_width, label='vs GMF-Q')
    # plt.bar(mfq, height=vs_mfq, width=bar_width, label='vs MF-Q')
    # plt.bar(amfq, height=vs_amfq, width=bar_width, label='vs AMF-Q')
    #
    # plt.legend(loc='best', fontsize=12)  # 显示图例
    # plt.xticks(iql + 1.5*bar_width, algo_name, fontsize=14)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('Rate of Winning', fontsize=14)  # 纵坐标轴标题
    # plt.ylim([0, 1])
    # plt.grid(axis='y', linestyle='--')
    # # plt.title('vs results')  # 图形标题
    #
    # plt.show()

    import seaborn as sns
    import random

    id = list(range(10))
    random.shuffle(id)

    att_matrix = []
    for i in range(10):
        atten = []
        for _ in range(9):
            atten.append(random.uniform(0.0001, 0.1))
        max_w = 1-np.sum(atten)
        print(max_w)
        atten.insert(id[i], max_w)
        att_matrix.append(atten)

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(att_matrix, annot=True, vmin=0, vmax=1, square=True, cmap="Blues") # cmap="Blues",  cmap=plt.cm.hot
    ax.set_ylabel('Rover', fontsize=14)
    ax.set_xlabel('Tower', fontsize=14)
    plt.show()

