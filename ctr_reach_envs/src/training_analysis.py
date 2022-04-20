import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, rc={'text.usetex' : True})
sns.set_style('white')


def process_files_and_get_dataframes(files, experiment_names):
    dfs = []
    for f, name in zip(files, names):
        df = pd.read_csv(f + '/progress.csv')
        df["experiment"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_errors_and_success_rate(df, labels):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    sns.lineplot(x='total/steps', y='errors', data=df, hue='experiment', legend=False, ax=ax1, style='experiment')
    sns.lineplot(x='total/steps', y='success rate', data=df, hue='experiment', legend=False, ax=ax2, style='experiment')
    plt.legend(title='Experiment', loc='center right', labels=labels)
    plt.xlabel("Training steps")
    ax1.set_ylabel("Errors (m)")
    ax2.set_ylabel("Success rate")
    plt.show()


if __name__ == '__main__':
    # Load in progress.csv file with data to plot for each experiment
    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
    #names = ['constrain_rotation/tro_constrain_0', 'free_rotation/tro_free_0']
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    names = ['two_tubes/tro_two_systems_2', 'three_tubes/tro_three_systems_0', 'four_tubes/tro_four_systems_0']
    files = [project_folder + names[0], project_folder + names[1], project_folder + names[2]]
    proc_df = process_files_and_get_dataframes(files, names)

    plot_error_and_success = True
    plot_goal_tolerance = False
    if plot_error_and_success:
        plot_errors_and_success_rate(proc_df, ['two tube', 'three tube', 'four tube'])
    if plot_goal_tolerance:
        sns.lineplot(x='total/steps', y='rollout/goal_tolerance', data=proc_df, hue='experimen')
        plt.show()

