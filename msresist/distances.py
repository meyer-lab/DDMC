"""Creates plots to visualize cell clustering data"""
import glob
import math
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.stats import RipleysKEstimator


def PlotSingleDistances(folder, extension, ax, log=False):
    """Plots boxplot of distance to closest cell give ImageJ data, with or without log transformation"""
    times = GetTimes(folder, extension)
    file_frame = pd.concat(Generate_dfs(folder, extension, times))
    if log:
        logs = []
        for length in file_frame["Length"]:
            logs.append(math.log(length))
        file_frame["Log Lengths"] = logs
        sns.boxplot(x="Time", y="Log Lengths", data=file_frame, ax=ax)
        ax.set_ylim(-1.7, 1.6)
        ax.set_title(folder)
    else:
        sns.boxplot(x="Time", y="Length", data=file_frame, ax=ax)
        ax.set_ylim(0, 4.5)
        ax.set_title(folder)


def GetTimes(folder, extension):
    """Takes in a folder and extension in correct format and returns list of times"""
    filenames = glob.glob("msresist/data/Distances/" + folder + "/Results_" + extension + "*.csv")
    filename_prefix = "msresist/data/Distances/" + folder + "/Results_" + extension + "_"
    filename_suffix = ".csv"
    times = []
    for file in filenames:
        time = int(file[len(filename_prefix): -len(filename_suffix)])
        times.append(time)
    times = sorted(times)
    return times


def Generate_dfs(folder, extension, times):
    """Generates dfs of the data at each time point with an added column for time"""
    file_list = []
    for time in times:
        file = pd.read_csv("msresist/data/Distances/" + folder + "/Results_" + extension + "_" + str(time) + ".csv")
        file["Time"] = time
        file_list.append(file)
    return file_list


def Calculate_closest(file_list, n=(1, 3)):
    """Calculates distances to nearby cells and returns as dataframe ready to plot"""
    distances_by_time = []
    for idx, file in enumerate(file_list):
        distances_df = pd.DataFrame()
        points = file.loc[:, "X":"Y"]
        # shortest_n_distances_lists = []
        shortest_n_distances = []
        for origin_cell in np.arange(points.shape[0]):
            distances = []
            x1, y1 = points.iloc[origin_cell, :]
            for other_cell in np.arange(points.shape[0]):
                x2, y2 = points.iloc[other_cell, :]
                distance = abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                distances.append(distance)
            distances = sorted(distances)
            # shortest_n_distances_lists.append(distances[1:(n+1)])
            shortest_n_distances.extend(distances[n[0]: (n[1] + 1)])
        distances_df["Distances"] = shortest_n_distances
        distances_df["Time"] = idx * 3
        distances_by_time.append(distances_df)
    return pd.concat(distances_by_time)


def PlotClosestN(folder, extension, ax, log=False, cells=(1, 3)):
    """Plots specified range of nearby cells as boxplots with or without log transformation"""
    times = GetTimes(folder, extension)
    file_list = Generate_dfs(folder, extension, times)
    plotting_frame = Calculate_closest(file_list, cells)
    if log:
        logs = []
        for length in plotting_frame["Distances"]:
            logs.append(math.log(length))
        plotting_frame["Log Distances"] = logs
        sns.boxplot(x="Time", y="Log Distances", data=plotting_frame, ax=ax)
        ax.set_ylim(-2, 2.3)
        ax.set_title(extension)
    else:
        sns.boxplot(x="Time", y="Distances", data=plotting_frame, ax=ax)
        ax.set_ylim(0, 8)
        ax.set_title(extension)


def PlotNhrsdistances(folder, mutants, treatments, replicates, ax, log=False, logmean=False, cells=(1, 3)):
    """Creates either raw/log grouped boxplot or log pointplot for distances to cells depending on log and logmean variables for range of nearby cells"""
    to_plot = Distances_import(folder, mutants, treatments, replicates, cells, logbool=False)
    if log:
        logs = []
        for length in to_plot["Distances"]:
            logs.append(math.log(length))
        to_plot["Log Distances"] = logs
        if logmean:
            sns.pointplot(x="Mutant", y="Log Distances", hue="Condition", data=to_plot, ci=68, join=False, dodge=0.4, ax=ax)
        else:
            sns.boxplot(x="Mutant", y="Log Distances", hue="Condition", data=to_plot, ax=ax)
    else:
        sns.boxplot(x="Mutant", y="Distances", hue="Condition", data=to_plot, ax=ax)
        ax.set_ylim(0, 5)
        # b = sns.swarmplot(x='Mutant', y='Distances', hue='Condition', data=to_plot, dodge=True, ax=ax)


def calculatedistances(file, mutant, treatment, replicate, cells=(1, 3)):
    """Calculates distances to range of other cells for a given mutant, treatment, and condition"""
    distances_df = pd.DataFrame()
    shortest_n_distances, _ = shortest_distances(file, cells)
    distances_df["Distances"] = shortest_n_distances
    distances_df["Mutant"] = mutant
    if replicate != 1:
        distances_df["Condition"] = treatment + str(replicate)
    else:
        distances_df["Condition"] = treatment
    return distances_df


def Plot_Logmean(folder, mutants, treatments, replicates, ax, vs_count=False, cells=(1, 3)):
    """Plots the log mean distance to neighbors by mutant and condition as given by cells argument.
    Will plot vs number of cells in image if vs_count is True"""
    to_plot = Distances_import(folder, mutants, treatments, replicates, cells, logbool=True, count_bool=vs_count)
    if vs_count:
        sns.scatterplot(x="Cells", y="Log_Mean_Distances", hue="Condition", style="Condition", data=to_plot, ax=ax)
        # for line in range(0, to_plot.shape[0]):
        # b.text(to_plot.Cells.iloc[line], to_plot.Log_Mean_Distances.iloc[line], to_plot.Mutant.iloc[line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    else:
        sns.pointplot(x="Mutant", y="Log_Mean_Distances", hue="Condition", data=to_plot, ci=68, join=False, dodge=0.25, ax=ax)


def calculatedistances_logmean(file, mutant, treatment, vs_count, cells=(1, 3)):
    """Calculates the average log distance to neighbors as defined by the cells argument and returns as a DataFrame in proper plotting format"""
    shortest_n_distances, count = shortest_distances(file, cells)
    logs = []
    for length in shortest_n_distances:
        logs.append(math.log(length))
    if vs_count:
        distances_df = {"Log_Mean_Distances": [np.mean(logs)], "Cells": [count], "Condition": [treatment], "Mutant": [mutant]}
    else:
        distances_df = {"Log_Mean_Distances": [np.mean(logs)], "Mutant": [mutant], "Condition": [treatment]}
    distances_df = pd.DataFrame(distances_df)
    return distances_df


def Distances_import(folder_name, mutant_list, treatment_list, replicate_number, cell_tuple, logbool, count_bool=None):
    """Imports specific files for the distance based plots, calculates distances, and returns a plottable df of distances"""
    dfs = []
    for mutant in mutant_list:
        mut_frames = []
        for treatment in treatment_list:
            for replicate in range(1, replicate_number + 1):
                if replicate != 1:
                    file = pd.read_csv("msresist/data/Distances/" + folder_name + "/Results_" + mutant + treatment + str(replicate) + ".csv")
                else:
                    file = pd.read_csv("msresist/data/Distances/" + folder_name + "/Results_" + mutant + treatment + ".csv")
                if logbool:
                    distances = calculatedistances_logmean(file, mutant, treatment, count_bool, cell_tuple)
                else:
                    distances = calculatedistances(file, mutant, treatment, replicate, cell_tuple)
                mut_frames.append(distances)
        mut_frame = pd.concat(mut_frames)
        dfs.append(mut_frame)
    to_plot = pd.concat(dfs)
    return to_plot


def shortest_distances(file_df, cell_tuple):
    """calculates distances for a specific file and cell neighbors set"""
    points = file_df.loc[:, "X":"Y"]
    shortest_n_distances = []
    for origin_cell in np.arange(points.shape[0]):
        distances = []
        x1, y1 = points.iloc[origin_cell, :]
        for other_cell in np.arange(points.shape[0]):
            x2, y2 = points.iloc[other_cell, :]
            distance = abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            distances.append(distance)
        distances = sorted(distances)
        shortest_n_distances.extend(distances[cell_tuple[0]: (cell_tuple[1] + 1)])
    return shortest_n_distances, points.shape[0]


def PlotRipleysK(folder, mutant, treatments, replicates, ax):
    """Plots the Ripley's K Estimate in comparison to the Poisson for a range of radii"""
    Kest = RipleysKEstimator(area=158.8761, x_max=14.67, y_max=10.83, x_min=0, y_min=0)
    r = np.linspace(0, 5, 51)
    r_for_df = r
    poisson = Kest.poisson(r)
    poisson_for_df = poisson
    # Done 5 times to create total of 6 to match 6 replicates
    for _ in range(5):
        poisson_for_df = np.hstack((poisson_for_df, poisson))
        r_for_df = np.hstack((r_for_df, r))
    data = np.vstack((r_for_df, poisson_for_df))
    for treatment in treatments:
        reps = ripleys_import(replicates, folder, mutant, treatment)
        treat_array = treat_array_func(reps, Kest, r, poisson)
        data = np.vstack((data, treat_array))
    df = pd.DataFrame(data).T
    df.columns = ["Radii", "Poisson", "Untreated", "Erlotinib", "AF154 + Erlotinib"]
    df = pd.melt(df, ["Radii"])
    df.columns = ["Radii", "Condition", "K Estimate"]
    sns.lineplot(x="Radii", y="K Estimate", hue="Condition", data=df, ci=68, ax=ax)
    ax.set_title(mutant)


def BarPlotRipleysK(folder, mutants, xticklabels, treatments, legendlabels, replicates, r):
    """Plots a bar graph of the Ripley's K Estimate values for all mutants and conditions in comparison to the Poisson at a discrete radius.
    Note that radius needs to be input as a 1D array for the RipleysKEstimator to work"""
    Kest = RipleysKEstimator(area=158.8761, x_max=14.67, y_max=10.83, x_min=0, y_min=0)
    poisson = Kest.poisson(r)
    mutant_dfs = []
    for z, mutant in enumerate(mutants):
        for j, treatment in enumerate(treatments):
            reps = ripleys_import(replicates, folder, mutant, treatment)
            treat_array = treat_array_func(reps, Kest, r, poisson, Kestbool=True)
            df = pd.DataFrame(treat_array)
            df.columns = ["K Estimate"]
            df["AXL mutants Y->F"] = xticklabels[z]
            df["Treatment"] = legendlabels[j]
            # add_poisson(poisson, mutant, df)
            mutant_dfs.append(df)
    df = pd.concat(mutant_dfs)
    b = sns.barplot(x="AXL mutants Y->F", y="K Estimate", hue="Treatment", data=df, ci=68)
    b.set_title("Radius of " + str(r[0]) + " Normalized to Poisson")
    b.legend(loc=2)


def BarPlotRipleysK_TimePlots(folder, mutant, extensions, treatments, r, ax):
    """Plots a bar graph of the Ripley's K Estimate values for one mutant in all conditions in comparison to the Poisson at a discrete radius.
    Note that radius needs to be input as a 1D array for the RipleysKEstimator to work"""
    Kest = RipleysKEstimator(area=158.8761, x_max=14.67, y_max=10.83, x_min=0, y_min=0)
    poisson = Kest.poisson(r)
    treatment_dfs = []
    for idx, extension in enumerate(extensions):
        file = pd.read_csv("msresist/data/Distances/" + folder + "/Results_" + extension + ".csv")
        points = file.loc[:, "X":"Y"].values
        treat_array = Kest(data=points, radii=r, mode="ripley") / poisson
        df = pd.DataFrame(treat_array)
        df.columns = ["K Estimate"]
        df["Mutant"] = mutant
        df["Treatment"] = treatments[idx]
        # add_poisson(poisson, mutant, df)
        treatment_dfs.append(df)
    df = pd.concat(treatment_dfs)
    sns.barplot(x="Mutant", y="K Estimate", hue="Treatment", data=df, ci=68, ax=ax)


def DataFrameRipleysK(folder, mutants, treatments, replicates, r):
    """Returns a DataFrame of the Ripleys K data along with poisson information"""
    Kest = RipleysKEstimator(area=158.8761, x_max=14.67, y_max=10.83, x_min=0, y_min=0)
    poisson = Kest.poisson(r)
    mutant_dfs = []
    for mutant in mutants:
        for treatment in treatments:
            reps = ripleys_import(replicates, folder, mutant, treatment)
            treat_array = treat_array_func(reps, Kest, r, poisson, Kestbool=True)
            df = pd.DataFrame(treat_array)
            df.columns = ["K Estimate"]
            df["Mutant"] = mutant
            df["Treatment"] = treatment
            # add_poisson(poisson, mutant, df)
            mutant_dfs.append(df)
    df = pd.concat(mutant_dfs)
    return df.groupby(["Mutant", "Treatment"]).mean()


def PlotRipleysK_TimeCourse(folder, extensions, timepoint, ax):
    """Plots the Ripley's K Estimate for a series of images over time by condition, compared to the Poisson."""
    r = np.linspace(0, 5, 51)
    Kest = RipleysKEstimator(area=158.8761, x_max=14.67, y_max=10.83, x_min=0, y_min=0)
    poisson = Kest.poisson(r)
    data = np.vstack((r, poisson))
    treatments = []
    for extension in extensions:
        file = pd.read_csv("msresist/data/Distances/" + folder + "/Results_" + extension + "_" + str(timepoint) + ".csv")
        points = file.loc[:, "X":"Y"].values
        treatments.append(points)
    Kests = []
    for point_set in treatments:
        Kests.append(Kest(data=point_set, radii=r, mode="ripley"))
    treat_array = np.vstack((Kests[0], Kests[1]))
    treat_array = np.vstack((treat_array, Kests[2]))
    data = np.vstack((data, treat_array))
    df = pd.DataFrame(data).T
    df.columns = ["Radii", "Poisson", "Untreated", "Erlotinib", "Erlotinib + AF154"]
    df = pd.melt(df, ["Radii"])
    df.columns = ["Radii", "Condition", "K Estimate"]
    sns.lineplot(x="Radii", y="K Estimate", hue="Condition", data=df, ci=68, ax=ax)
    ax.set_title(str(timepoint) + " hours")


def ripleys_import(replicate_number, folder_name, mutant_name, treatment_name):
    """Imports replicates of the cell locations for analysis by Ripley's K function"""
    reps = []
    for replicate in range(1, replicate_number + 1):
        if replicate != 1:
            file = pd.read_csv("msresist/data/Distances/" + folder_name + "/Results_" + mutant_name + treatment_name + str(replicate) + ".csv")
        else:
            file = pd.read_csv("msresist/data/Distances/" + folder_name + "/Results_" + mutant_name + treatment_name + ".csv")
        points = file.loc[:, "X":"Y"].values
        reps.append(points)
    return reps


def treat_array_func(rep_list, Kest_func, radius, poisson_val, Kestbool=False):
    """Applies the Ripley's K function and returns the resulting values as an array"""
    Kests = []
    if Kestbool:
        for point_set in rep_list:
            Kests.append(Kest_func(data=point_set, radii=radius, mode="ripley") / poisson_val)
        if len(Kests) < 2:
            treat_array = Kests[0]
            return treat_array
    else:
        for point_set in rep_list:
            Kests.append(Kest_func(data=point_set, radii=radius, mode="ripley"))
    treat_array = np.hstack((Kests[0], Kests[1]))
    for i in range(2, len(Kests)):
        treat_array = np.hstack((treat_array, Kests[i]))
    return treat_array


def add_poisson(poisson_val, mutant_name, dataframe):
    """Optionally adds poisson data as comparison for bar graphs"""
    poisson_df = pd.DataFrame(poisson_val)
    poisson_df.columns = ["K Estimate"]
    poisson_df["Mutant"] = mutant_name
    poisson_df["Treatment"] = "Poisson"
    dataframe = pd.concat([poisson_df, dataframe])
    return dataframe
