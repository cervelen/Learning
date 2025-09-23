#include "perceptron.h"
#include <vector>
#include <iostream>
#include <fstream>


        Perceptron::Perceptron(int inputsize, double learningrate){
            for (int i=0; i < inputsize; i++){
                weights.push_back(0.0);
            }
            this->learningrate = learningrate;
            bias = 0.0;
        }

        int Perceptron::predict(const std::vector<double>& inputs){
            double sum = bias;
            for (int i = 0; i < inputs.size(); i++){
                sum += (weights[i] * inputs[i]);
            }
            if (sum >= 0){
                return 1;
            }
            else{
                return -1;
            }
        }

        void Perceptron::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<int>& labels, int& epochs){
            for (int k = 0; k < epochs; k++){
                for (int i = 0; i < trainingdata.size(); i++){
                    int prediction = 0;
                    double sum = bias;
                    for (int j = 0; j < trainingdata[i].size(); j++){
                        sum += (weights[j] * trainingdata[i][j]);
                    }
                    if (sum >= 0){
                        prediction = 1;
                    }
                    else{
                        prediction = -1;
                    }

                    if (prediction != labels[i]){
                        for (int j = 0; j < weights.size(); j++){
                            weights[j] = weights[j] + (learningrate * labels[i] * trainingdata[i][j]);
                        }
                        bias = bias + (learningrate * labels[i]);
                    }
                }
            }
        }



int main(){
    // Define training data for AND logic gate
    std::vector<std::vector<double>> training_data = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    // Expected outputs: AND gate
    std::vector<int> labels = {-1, -1, -1, 1}; // Perceptron expects -1 and 1 (not 0 and 1)

    int input_size = 2;
    double learning_rate = 0.1;
    int epochs = 3;

    // Create perceptron
    Perceptron p(input_size, learning_rate);

    // Train it
    p.train(training_data, labels, epochs);

    // Test it
    std::cout << "Predictions after training:\n";
    for (const auto& sample : training_data) {
        int result = p.predict(sample);
        std::cout << sample[0] << " AND " << sample[1] << " = " << ((result == 1) ? 1 : 0) << std::endl;
    }
}