## Preparing Datasets for HandsOnVLM

1. Download the assets from [Google Drive](https://drive.google.com/file/d/1LTFgLfCR653khkouaQRVtAw_OuEkNUgo/view?usp=sharing) and place the `data` directory at `hoi_forecast/data` and `common` at `hoi_forcast/common`.
2. Follow the [instructions](https://github.com/NVlabs/LITA/blob/main/docs/Video_Data.md) to prepare datasets in LITA.
3. Download EPIC-KITCHEN using this [repo](https://github.com/epic-kitchens/epic-kitchens-download-scripts) with command `python epic_downloader.py --rgb-frames`.
4. Create a hard link from the EPIC download dir to hoi-forecast data. `cd ./hoi_forecast/data && ln -s <path_to_epic_kitchen>/EPIC-KITCHENS/ ./`