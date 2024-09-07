import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_contour, plot_edf, plot_intermediate_values, plot_param_importances, plot_rank, plot_slice, plot_timeline
import optuna.visualization as vis
import plotly.io as pio

storage = optuna.storages.RDBStorage('sqlite:///example.db')

studies = optuna.study.get_all_study_summaries(storage=storage)

# choose study
#study_name = "random_sampler_40"
study_name = studies[1].study_name
study_name = "bayesian_300"

# load study
study = optuna.load_study(study_name=study_name, storage=storage)

path_for_plots = '/.automount/home/home__home1/institut_3a/seiler/HHbbgg_conditional_classifiers/models/test_hyperplots/'

#plots
fig = plot_optimization_history(study)
pio.write_image(fig, f'{path_for_plots}/history_plot.png', scale=2)
fig = plot_parallel_coordinate(study)
pio.write_image(fig, f'{path_for_plots}/parallel_coordinate_plot.png', scale=2)
fig = plot_intermediate_values(study)
pio.write_image(fig, f'{path_for_plots}/intermediate_values_plot.png', scale=2)
fig = plot_param_importances(study)
pio.write_image(fig, f'{path_for_plots}/param_importances_plot.png', scale=2)

fig = vis.plot_contour(study)
fig.update_layout(
    font=dict(size=10),   # Schriftgröße für alle Texte im Plot
    xaxis_tickangle=-45,  # Neige die x-Achsen-Tick-Labels für bessere Lesbarkeit
    yaxis_tickangle=0,  # Stelle sicher, dass die y-Achsen-Tick-Labels nicht geneigt sind
    height=800,  # Erhöhe die Plot-Höhe, um vertikalen Platz zu schaffen
    autosize=False,  # Deaktiviere die automatische Größenanpassung, falls du eine feste Größe möchtest
    width=1200  # Breite des Plots, um horizontalen Platz zu schaffen
)
pio.write_image(fig, f'{path_for_plots}/contour_plot.png', scale=2)

fig = plot_slice(study)
pio.write_image(fig, f'{path_for_plots}/slice_plot.png', scale=2)

fig = vis.plot_rank(study)
fig.update_layout(
    font=dict(size=10),   # Schriftgröße für alle Texte im Plot
    xaxis_tickangle=-45,  # Neige die x-Achsen-Tick-Labels für bessere Lesbarkeit
    yaxis_tickangle=0,  # Stelle sicher, dass die y-Achsen-Tick-Labels nicht geneigt sind
    height=800,  # Erhöhe die Plot-Höhe, um vertikalen Platz zu schaffen
    autosize=False,  # Deaktiviere die automatische Größenanpassung, falls du eine feste Größe möchtest
    width=1200  # Breite des Plots, um horizontalen Platz zu schaffen
)
pio.write_image(fig, f'{path_for_plots}/rank_plot.png', scale=2)

fig = plot_edf(study)
pio.write_image(fig, f'{path_for_plots}/edf_plot.png', scale=2)
fig = plot_timeline(study)
pio.write_image(fig, f'{path_for_plots}/timeline_plot.png', scale=2)