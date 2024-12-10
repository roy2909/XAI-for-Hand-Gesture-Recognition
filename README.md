# Explainable AI for Hand Gesture Recognition
* Author Rahul Roy
* [Portfolio Post](https://roy2909.github.io/Transfer/)  
&nbsp;
* Based on the [paper](https://www.sciencedirect.com/science/article/pii/S0020025524005802#fm0050)
* Dataset recorded by [Suguru Kanoga](https://github.com/Suguru55/Wearable_Sensor_Long-term_sEMG_Dataset)
* Previous work by [Sonia Yuxiao Lai](https://github.com/aonai/long_term_EMG_myo/tree/main)

![shapley](https://github.com/user-attachments/assets/a6fb1504-26a3-4517-81b8-e59d890f6cae)

### Overview
This project aims to improve upon a previously developed hand gesture recognition [system](https://github.com/aonai/long_term_EMG_myo/tree/main) by incorporating Explainable AI techniques. The goal is to enhance the model's interpretability and transparency. 

### Usage
##### 0. Prepare Data
i. Specify the following file locations:
* dataset: `data_dir`
* processed dataset: `processed_data_dir`
* trained model weights: `path_base_weights`, `path_DANN_weights`, `path_SCADANN_weights`
* model test results: `path_result` 

ii. Process raw datasets into 
* formatted example arrays with 7 features (proposed by [Rami N. Khushaba](https://github.com/RamiKhushaba/getTSDfeat))
`read_data_training(path=data_dir, store_path=processed_data_dir)`
* or formatted spectrograms of shape (4, 8, 10)   
`read_data_training(path=data_dir, store_path=processed_data_dir, spectrogram=True)`   

iii. Load processed examples and labels   
```
with open(processed_data_dir + "/training_session.pickle", 'rb') as f:
    dataset_training = pickle.load(file=f)
examples_datasets_train = dataset_trainin['examples_training']
labels_datasets_train = dataset_training['labels_training']
```  

iv. Specify the following training parameters: 
* `num_kernels`: list of integers where each integer corresponds to number of kernels of each layer for the base model
* `filter_size`:  a 2d list of shape (m, 2), where m is number of levels and each list corresponds to kernel size of the base ConvNet model
* `number_of_classes`: total number of gestures recorded
* `number_of_cycles_total`: total number of trails in one session
* `feature_vector_input_length`: length of one formatted example (252) for the base TSD model
* `batch_size` and `learning_rate` 


##### 1. Base Model  
i. Train a base model using Convolutional Network (ConvNet) with `neural_net='ConvNet'`  
```
train_fine_tuning(examples_datasets_train, labels_datasets_train,
            num_kernels=<num_kernels_of_each_layer>,   
            path_weight_to_save_to=<path_base_weights>,  
            number_of_classes=<number_of_classes>,   
            number_of_cycles_total=<number_of_cycles_total>,
            batch_size=<batch_size>,  
            feature_vector_input_length=<length_of_formatted_example>,
            learning_rate=<learning_rate>,  
            neural_net=<choise_of_model>,
            filter_size=<kernel_size_of_ConvNet>)
```
Follow the steps in `shapley_feedback.ipynb` to train a base model using Shapley Feedback for base model.

ii. Test and record results of the base model with and without fine-tuning
```
test_standard_model_on_training_sessions(examples_datasets_train, labels_datasets_train,
                                num_neurons=<num_kernels_of_each_layer>,  
                                use_only_first_training=<whether_to_use_fine_tuned_model>,
                                path_weights=<path_base_weights>,
                                save_path=<path_result>,   
                                algo_name=<result_file_name>,
                                number_of_cycles_total=<number_of_cycles_total>,  
                                number_of_classes=<number_of_classes>,  
                                cycle_for_test=<testing_trial_num>,
                                neural_net=<choise_of_model>,
                                filter_size=<kernel_size_of_ConvNet>)
```                             
##### 2. Domain-Adversarial Neural Network (DANN) without Shapley Feedback
i. Train a DANN model from the base model
```
train_DANN(examples_datasets_train, labels_datasets_train, 
        num_kernels=<num_kernels_of_each_layer>,
        path_weights_fine_tuning=<path_base_weights>,
        number_of_classes=<number_of_classes>,
        number_of_cycles_total=<number_of_cycles_total>,
        batch_size=<batch_size>,
        path_weights_to_save_to=<path_DANN_weights>, 
        learning_rate=<learning_rate>,
        neural_net=<choise_of_model>,
        filter_size=<kernel_size_of_ConvNet>)
```
ii. Test and record results of DANN model
```
test_DANN_on_training_sessions(examples_datasets_train, labels_datasets_train,
                            num_neurons=<num_kernels_of_each_layer>,  
                            path_weights_DA=<path_DANN_weights>,
                            algo_name=<result_file_name>, 
                            save_path=<path_result>, 
                            number_of_cycles_total=<number_of_cycles_total>,
                            path_weights_normal=<path_base_weights>, 
                            number_of_classes=<number_of_classes>,
                            cycle_for_test=<testing_trial_num>, 
                            neural_net=<choise_of_model>,
                            filter_size=<kernel_size_of_ConvNet>)
```
##### 3. Domain-Adversarial Neural Network (DANN) with Shapley Feedback

i. Train a DANN model from the base model with Shapley Feedback
```train_improved_DANN(examples_datasets_train, labels_datasets_train, 
        num_kernels=<num_kernels_of_each_layer>,
        path_weights_fine_tuning=<path_base_weights>,
        number_of_classes=<number_of_classes>,
        number_of_cycles_total=<number_of_cycles_total>,
        batch_size=<batch_size>,
        path_weights_to_save_to=<path_SCADANN_weights>, 
        learning_rate=<learning_rate>,
        neural_net=<choise_of_model>,
        filter_size=<kernel_size_of_ConvNet>)
```
ii. Test and record results of DANN model with Shapley Feedback
```
test_improved_DANN_on_training_sessions(examples_datasets_train, labels_datasets_train,
                            num_neurons=<num_kernels_of_each_layer>,  
                            path_weights_DA=<path_SCADANN_weights>,
                            algo_name=<result_file_name>, 
                            save_path=<path_result>, 
                            number_of_cycles_total=<number_of_cycles_total>,
                            path_weights_normal=<path_base_weights>, 
                            number_of_classes=<number_of_classes>,
                            cycle_for_test=<testing_trial_num>, 
                            neural_net=<choise_of_model>,
                            filter_size=<kernel_size_of_ConvNet>)
```
### Results
