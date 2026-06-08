import json 
from ptls.make_datasets_spark import DatasetConverter


if __name__ == '__main__':
    dataset_converter = DatasetConverter()
    dataset_converter.run()

    for col in dataset_converter.encoders:
        # create a mapping from col to _orig_<col>
        mapping = dict()
        for row in dataset_converter.encoders[col].collect():
            original = row[f'_orig_{col}']
            encoded = row[col]

            mapping[encoded] = original

        with open(f'./data/{col}_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=4)

    print(type(dataset_converter.encoders), dataset_converter.encoders)
    # print(type(dataset_converter.df_data), dataset_converter.df_data)
    # print(type(dataset_converter.features), dataset_converter.features)
    # print(type(dataset_converter.client_features), dataset_converter.client_features)
    # # print(dataset_converter.features.show(1))
    # print(dataset_converter.df_data_pre_encode.head(1))
    # print(dataset_converter.df_data_pos_encode.head(1))
    print('*' * 20)
