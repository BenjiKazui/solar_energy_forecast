import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_chronological(y, final_preds_df):
    fig, ax = plt.subplots()
    ax.plot(y.time, y.energy, label="Actual Energy")
    ax.plot(final_preds_df.time, final_preds_df.energy, label="Predicted Energy")

    # Set major ticks for the start of each year and minor ticks for each month
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Format major ticks to show only the year as a number
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xlabel("Time")
    plt.ylabel("Energy in MWh")
    plt.title("Actual/Predicted Generated Solar Energy in BW")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def help_plot_vertically(ax, z, z_label, window_size):
    unique_years = z["time"].dt.year.unique()

    for year in unique_years:
        # Filter data for a single year
        yearly_data = z[z["time"].dt.year == year].copy()
        
        # Use only energy data that has values != 0
        #yearly_data = yearly_data[yearly_data["energy"] != 0]

        # use a rolling mean with window size of 24*30 = 24h * 30 days = 1 Month
        if "energy" in z.columns:
            yearly_data[f"rolling_{window_size}h"] = yearly_data["energy"].rolling(window=window_size, min_periods=1).mean()
        elif "energy predictions" in z.columns:
            yearly_data[f"rolling_{window_size}h"] = yearly_data["energy predictions"].rolling(window=window_size, min_periods=1).mean()
        else:
            print("Couldn't find column 'energy' or 'energy predictions'")

        # Normalize the x-axis to only show months (ignore year)
        month_only = yearly_data["time"].dt.strftime("%m-%d")  # Format as 'MM-DD'

        # Plot each year separately
        #ax.scatter(month_only, yearly_data["rolling_24h"], label=f"{z_label} {year}", s=1)#, alpha=0.7, edgecolors="k")
        ax.plot(month_only, yearly_data[f"rolling_{window_size}h"], label=f"{z_label} {year}", linewidth=0.5)#, alpha=0.7, edgecolors="k")


def plot_vertically(data_type, data_list, label_list, window_size, save=False, save_path=None):

    if len(data_list) != len(label_list):
        print("Provide data_list and label_list with the corresponding data and labels, data_list and label_list must be same length.")
        return None
    
    fig, ax = plt.subplots()

    for i, data in enumerate(data_list):
        help_plot_vertically(ax, data, label_list[i], window_size)

    # Set x-axis format to show months only
    ax.set_xticks(["01-01", "02-01", "03-01", "04-01", "05-01", "06-01",
                   "07-01", "08-01", "09-01", "10-01", "11-01", "12-01"])
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    plt.xlabel("Month")
    plt.ylabel("Energy in MWh")
    plt.title(f"Solar Energy Generation ({data_type}) using window of size: {window_size} hours")
    plt.legend()  # Use the legend to differentiate years
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    if save == True and save_path != None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot-Vertical-Figure saved to: ", save_path)


def plot_zoomed_in_window(data_list, label_list, start_date, end_date, save=False, save_path=None):
    """
    Plot a zoomed-in window of the data between start_date and end_date.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, data in enumerate(data_list):
        data = data[(data["time"] >= start_date) & (data["time"] <= end_date)]
        print("data after filtering:\n", data.head())
        if "energy" in data.columns:
            ax.plot(data["time"], data["energy"], label=label_list[i], linewidth=0.5)
        elif "energy predictions" in data.columns:
            ax.plot(data["time"], data["energy predictions"], label=label_list[i], linewidth=0.5)
        else:
            print("Couldn't find column 'energy' or 'energy predictions'")

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy in MWh")
    ax.set_title(f"Zoomed-in view from {start_date} to {end_date}")
    ax.legend() 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Clean x-axis ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

    if save == True and save_path != None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Plot-Zoomed-In-Figure saved to: ", save_path)
