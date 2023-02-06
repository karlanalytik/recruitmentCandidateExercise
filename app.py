from model_1 import MediaModel
from shiny import App, reactive, render, ui


def panel_box(*args, **kwargs):
    return ui.div(
        ui.div(*args, class_="card-body"),
        **kwargs,
        class_="card mb-3",
    )


app_ui = ui.page_fluid(
    {"class": "p-4"},
    ui.row(
        ui.column(
            10,
            "Google Search Volumes Predictor",
            panel_box(
                ui.input_slider(
                    "strength", "Retention Factor", min=0, max=1, value=0.5, step=0.1
                ),
                ui.input_slider("lenght", "Weeks", min=0, max=5, value=3, step=1),
            ),
            ui.navset_tab_card(
                ui.nav(
                    "Model 1",
                    ui.column(
                        10,
                        "With the selected values we have the following fit graph:",
                        ui.output_plot("fit_plot", width="700px", height="500px"),
                    ), 
                    ui.column(
                        10,
                        "Results obtained in each fold during cross validation process:",
                        ui.output_table("crossval_table"),
                        ui.output_text_verbatim("txt_crossval") 
                    ),
                    ui.column(
                        10,
                        "Efficiency measure (ROI):",
                        ui.output_table("generate_roi_table"), 
                    ),
                    ui.column(
                        10,
                        "Campaign contribution to the search volume:", # No se ve
                        ui.output_plot("camp_contribution_plot", width="700px", height="500px"),
                    ),
                ),
                ui.nav(
                    "Model 2",
                    ui.column(
                        10,
                        #ui.output_plot("contr_plot", width="700px", height="500px"),
                    ),
                ),
            ),
        ),
        # ui.column(
        #     8,
        #     #ui.output_text_verbatim("txt"),
        #     ui.output_plot("fit_plot", width="500px", height="500px"),
        #     ui.output_table("generate_roi_table"),
        # ),
    ),
)


def server(input, output, session):
    
    @output
    @render.plot
    def fit_plot(): 
        medmod = MediaModel(input.strength(), input.lenght(), a = 0.5)
        df, X, y = medmod.prepare_data()
        coef, intercept, y_pred = medmod.predict(X, y)
        fig = medmod.make_fit_plot(X, y, y_pred)
        return fig

    @output
    @render.table
    def crossval_table():
        medmod = MediaModel(input.strength(), input.lenght(), a = 0.5)
        df, X, y = medmod.prepare_data()
        cv_df, cv_r2_mean, cv_n_1_r2_mean = medmod.get_crossval_table(X, y)
        return (
            cv_df.style.set_table_attributes(
                'class="dataframe shiny-table table w-auto"'
            ).hide(axis="index")
             .format(
                {
                    "fit_time": "{0:0.4f}",
                    "score_time": "{0:0.4f}",
                    "test_r2": "{0:0.4f}",
                    "test_neg_mean_squared_error": "{0:0.4f}"
                }
             )
        )

    
    @output
    @render.text
    def txt_crossval():
        medmod = MediaModel(input.strength(), input.lenght(), a = 0.5)
        df, X, y = medmod.prepare_data()
        cv_df, cv_r2_mean, cv_n_1_r2_mean = medmod.get_crossval_table(X, y)
        return f"The mean of the r2 is {cv_r2_mean} and if the last fold is discarded the mean is {cv_n_1_r2_mean}."

    
    @output
    @render.table
    def generate_roi_table(): 
        medmod = MediaModel(input.strength(), input.lenght(), a = 0.5)
        df, X, y = medmod.prepare_data()
        coef, intercept, y_pred = medmod.predict(X, y)
        adj_contributions = medmod.calculate_roi(X, y, coef, intercept)
        roi_df = medmod.roi_table(df, adj_contributions)
        return (
            roi_df.style.set_table_attributes(
                'class="dataframe shiny-table table w-auto"'
            ).hide(axis="index")
             .format(
                {
                    "campaign_n": "{0:0.0f}",
                    "searches_from_camp_n": "{0:0.2f}",
                    "spendings_on_camp_n": "{0:0.0f}",
                    "roi": "{0:0.3f}",
                    "1/roi": "{0:0.2f}",
                }
             )
        )


    @output
    @render.plot
    def camp_contribution_plot(): # No se ve
        medmod = MediaModel(input.strength(), input.lenght(), a = 0.5)
        df, X, y = medmod.prepare_data()
        coef, intercept, y_pred = medmod.predict(X, y)
        adj_contributions = medmod.calculate_roi(X, y, coef, intercept)
        fig2 = medmod.contribution_plot(adj_contributions)
        return fig2


app = App(app_ui, server, debug=True)