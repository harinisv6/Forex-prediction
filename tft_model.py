import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning import Trainer
from utils.data_loader import load_time_series_data

def train_tft():
    df = load_time_series_data()

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time",
        target="target",  # next day's return/direction
        group_ids=["currency_pair"],
        max_encoder_length=30,
        max_prediction_length=1,
        time_varying_known_reals=["time"],
        time_varying_unknown_reals=["open", "high", "low", "close", "volume"]
    )

    dataloader = dataset.to_dataloader(train=False, batch_size=32)

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.001,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        loss=RMSE()
    )

    trainer = Trainer(max_epochs=1, accelerator="cpu")  # use more epochs for real training
    # trainer.fit(tft)  # uncomment to train

    tft_pred = tft.predict(dataloader).numpy()
    return tft, tft_pred
