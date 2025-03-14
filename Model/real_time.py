class RealTimeNBAStatPredictor:
    def __init__(self, model_path, data_folder, cache_path):
        self.model = tf.keras.models.load_model(model_path)
        self.data_folder = data_folder
        self.cache_path = cache_path
        self.df = load_multi_season_data(data_folder, cache_path)

    def get_recent_form(self, player_name, last_n_games=10):
        """Retrieve rolling average of a player's last N games."""
        player_df = self.df[self.df['Player'] == player_name].tail(last_n_games)
        return player_df.mean()

    def adjust_for_real_time(self, base_prediction, recent_form, oppDfRtg):
        """Dynamically adjust predictions based on recent form & opponent rating."""
        adj_prediction = base_prediction.copy()

        # **Factor in opponent defensive rating**
        def_rtg_adjustment = np.clip((oppDfRtg - 110) / 10, -0.1, 0.1)  # Adjust by ±10%

        # **Apply rolling form adjustment**
        for stat in adj_prediction.keys():
            rolling_stat = recent_form.get(stat, 0)
            adj_prediction[stat] = (adj_prediction[stat] * 0.8) + (rolling_stat * 0.2)  # 80% Model, 20% Recent Form
            adj_prediction[stat] *= (1 + def_rtg_adjustment)  # Apply defense adjustment

        return adj_prediction

    def predict(self, player_name, oppDfRtg):
        """Predict a player's stats, adjusting for real-time data."""
        recent_form = self.get_recent_form(player_name)
        
        # **Prepare Input**
        new_game = {'oppDfRtg': oppDfRtg, **{col: 0 for col in self.df.columns if col.startswith('OPP_')}}
        new_game_scaled = scaler.transform([list(new_game.values())])
        new_game_tensor = np.array(new_game_scaled).reshape(1, 10, len(new_game))

        # **Run Model Prediction**
        base_prediction = self.model.predict(new_game_tensor)[0]
        stat_prediction = dict(zip(target_columns, base_prediction))

        # **Adjust for Real-Time Factors**
        adjusted_prediction = self.adjust_for_real_time(stat_prediction, recent_form, oppDfRtg)
        return adjusted_prediction
