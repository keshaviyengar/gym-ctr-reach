import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def process_files_and_get_dataframes(system_idx, experiments):

    files = []
    dfs = []
    for name in experiments:
        for id in system_idx:
            f = project_folder + name + "/evaluations_" + str(id) + ".csv"
            df = pd.read_csv(f)
            dg = np.array([df['desired_goal_x'], df['desired_goal_y'], df['desired_goal_z']])
            ag = np.array([df['achieved_goal_x'], df['achieved_goal_y'], df['achieved_goal_z']])
            sg = np.array([df['starting_position_x'], df['starting_position_y'], df['starting_position_z']])
            df['errors_pos'] = np.linalg.norm(np.transpose(ag - dg), axis=1) * 1000
            df['goal_dist'] = np.linalg.norm(np.transpose(sg - dg), axis=1) * 1000
            df['success'] = df['errors_pos'] < 1.0
            df["system"] = id
            df['name'] = name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_alpha_box_plots(df, alpha):
    sns.boxplot(x="alpha_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Alpha " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.ylim([0, 100])
    plt.show()

def plot_B_box_plots(df, alpha):
    sns.boxplot(x="B_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Beta " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.show()


if __name__ == '__main__':
    # Load in data
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    #name = ['four_systems/tro_four_systems_sample']
    name = ['three_systems/tro_three_systems_2']
    system_idx = [0]
    proc_df = process_files_and_get_dataframes(system_idx, name)

    # Summary of statistics
    for system in system_idx:
        system_df = proc_df[proc_df['system'] == system]
        print("mean errors: " + str(np.mean(system_df["errors_pos"])))
        print("std errors: " + str(np.std(system_df["errors_pos"])))
        print("success rate: " + str(system_df[system_df["errors_pos"] < 1].shape))

        plot_goal_distance_scatter = False
        plot_goal_distance_success = False
        plot_rot_joints_box_plot = False
        wrap_angles = True
        plot_ext_joints_box_plot = True
        plot_3d_achieved_workspace = False
        plot_3d_desired_workspace = False
        error_threshold = 2

        plot_polar_error = False

        if plot_polar_error:
            fig = plt.figure(figsize=(5, 5), dpi=100)
            sns.set(font_scale=2.0)
            sns.set_style(style='white')
            ax = fig.add_subplot(projection='polar', xlim=(-180, 180))
            ax.scatter(system_df['alpha_achieved_3'], system_df['errors_pos'], s=20, c=system_df['errors_pos'],
                       cmap='cividis',
                       vmin=0, vmax=35)
            ax.text(np.deg2rad(80), 25, 'Error (mm)',
                    rotation=90, ha='center', va='center', fontsize=17)
            ax.set_thetamin(-180.0)
            ax.set_thetamax(180.0)
            ax.set_rmax(100)
            ax.set_rticks([20, 40])
            ax.set_thetagrids(range(-180, 180, 45))
            plt.show()

        if plot_goal_distance_scatter:
            slope, intercept, r_value, p_value, std_err = stats.linregress(system_df['goal_dist'], system_df['errors_pos'])
            print("slope: ", slope)
            print("intercept: ", intercept)
            sns.regplot(x='goal_dist', y='errors_pos', data=system_df, ci=None, scatter_kws={"s": 10})
            plt.show()

        if plot_goal_distance_success:
            g = sns.jointplot(x="goal_dist", y="errors_pos", data=system_df, kind="hex")
            plt.show()

        num_bins = 10
        if plot_rot_joints_box_plot:
            # Rotation plots
            rotation_bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), num_bins)
            rot_label_bins = np.around(np.linspace(np.deg2rad(-180), np.deg2rad(180), num_bins - 1), 2)
            for alpha in range(1, 4):
                if wrap_angles:
                    system_df["alpha_achieved_" + str(alpha)] = (system_df["alpha_achieved_" + str(alpha)] + np.pi) % (
                            2 * np.pi) - np.pi
                system_df["alpha_achieved_" + str(alpha) + "_bins"] = pd.cut(system_df["alpha_achieved_" + str(alpha)],
                                                                           bins=rotation_bins, labels=rot_label_bins)
                plot_alpha_box_plots(system_df, alpha)

        if plot_ext_joints_box_plot:
            # Extension plots
            min_beta = np.min([system_df.B_achieved_1.min(), system_df.B_achieved_2.min(), system_df.B_achieved_3.min()])
            max_beta = np.max([system_df.B_achieved_1.max(), system_df.B_achieved_2.max(), system_df.B_achieved_3.max()])
            extension_bins = np.linspace(min_beta, max_beta, num_bins)
            ext_label_bins = np.around(np.linspace(min_beta, max_beta, num_bins - 1), 2)
            for beta in range(1, 4):
                system_df["B_achieved_" + str(beta) + "_bins"] = pd.cut(system_df["B_achieved_" + str(beta)],
                                                                      bins=extension_bins, labels=ext_label_bins)
                plot_B_box_plots(system_df, beta)

        # 3D workspace plots
        if plot_3d_achieved_workspace:
            fig = plt.figure(figsize=(10, 5), dpi=150)
            sns.set(font_scale=1.2)
            sns.set_style(style='white')
            ax3D = fig.add_subplot(1, 2, 1, projection='3d')
            ax3D.scatter(system_df['achieved_goal_x'] * 1000, system_df['achieved_goal_y'] * 1000,
                         system_df['achieved_goal_z'] * 1000,
                         c=system_df['errors_pos'], s=5, vmin=0, vmax=35, cmap='cividis')
            ax3D.set_xlabel("X (mm)")
            ax3D.set_ylabel("Y (mm)")
            ax3D.set_zlabel("Z (mm)")
            axis_lims = 0.200 * 1000
            ax3D.set_xlim3d([-axis_lims, axis_lims])
            ax3D.set_xticks([-50, 0, 50])
            ax3D.set_ylim3d([-axis_lims, axis_lims])
            ax3D.set_yticks([-50, 0, 50])
            ax3D.set_zlim3d([0.0, 2 * axis_lims])
            ax3D.set_zticks([0, 50, 100, 150])
            ax3D.set_title("All Errors")

            ax3D = fig.add_subplot(1, 2, 2, projection='3d')
            system_df_tol = system_df[system_df['errors_pos'] > error_threshold]
            p = ax3D.scatter(system_df_tol['achieved_goal_x'] * 1000, system_df_tol['achieved_goal_y'] * 1000,
                             system_df_tol['achieved_goal_z'] * 1000,
                             c=system_df_tol['errors_pos'], s=20, vmin=0, vmax=35, cmap='cividis')
            ax3D.set_xlabel("X (mm)")
            ax3D.set_ylabel("Y (mm)")
            ax3D.set_zlabel("Z (mm)")
            ax3D.set_xlim3d([-axis_lims, axis_lims])
            ax3D.set_xticks([-50, 0, 50])
            ax3D.set_ylim3d([-axis_lims, axis_lims])
            ax3D.set_yticks([-50, 0, 50])
            ax3D.set_zlim3d([0.0, 2 * axis_lims])
            ax3D.set_zticks([0, 50, 100, 150])
            ax3D.set_title("Errors > " + str(error_threshold) + " mm")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
            fig.colorbar(p, cax=cbar_ax)

            plt.show()
        if plot_3d_desired_workspace:
            fig = plt.figure(figsize=(10, 5), dpi=150)
            sns.set(font_scale=1.2)
            sns.set_style(style='white')
            ax3D = fig.add_subplot(1, 2, 1, projection='3d')
            ax3D.scatter(system_df['desired_goal_x'] * 1000, system_df['desired_goal_y'] * 1000,
                         system_df['desired_goal_z'] * 1000,
                         c=system_df['errors_pos'], s=5, vmin=0, vmax=35, cmap='cividis')
            ax3D.set_xlabel("X (mm)")
            ax3D.set_ylabel("Y (mm)")
            ax3D.set_zlabel("Z (mm)")
            axis_lims = 0.200 * 1000
            ax3D.set_xlim3d([-axis_lims, axis_lims])
            ax3D.set_xticks([-50, 0, 50])
            ax3D.set_ylim3d([-axis_lims, axis_lims])
            ax3D.set_yticks([-50, 0, 50])
            ax3D.set_zlim3d([0.0, 2 * axis_lims])
            ax3D.set_zticks([0, 50, 100, 150])
            ax3D.set_title("All Errors")

            ax3D = fig.add_subplot(1, 2, 2, projection='3d')
            system_df_tol = system_df[system_df['errors_pos'] > error_threshold]
            p = ax3D.scatter(system_df_tol['desired_goal_x'] * 1000, system_df_tol['desired_goal_y'] * 1000,
                             system_df_tol['desired_goal_z'] * 1000,
                             c=system_df_tol['errors_pos'], s=20, vmin=0, vmax=35, cmap='cividis')
            ax3D.set_xlabel("X (mm)")
            ax3D.set_ylabel("Y (mm)")
            ax3D.set_zlabel("Z (mm)")
            ax3D.set_xlim3d([-axis_lims, axis_lims])
            ax3D.set_xticks([-50, 0, 50])
            ax3D.set_ylim3d([-axis_lims, axis_lims])
            ax3D.set_yticks([-50, 0, 50])
            ax3D.set_zlim3d([0.0, 2 * axis_lims])
            ax3D.set_zticks([0, 50, 100, 150])
            ax3D.set_title("Errors > " + str(error_threshold) + " mm")
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
            fig.colorbar(p, cax=cbar_ax)

            plt.show()
