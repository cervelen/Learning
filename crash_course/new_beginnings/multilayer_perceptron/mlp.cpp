#include "mlp.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>
#include <cmath>

        std::vector<double> softmax(std::vector<double> inputvec){
            std::vector<double> outputvec;
            double denominator = 0;
            for (int i = 0; i < inputvec.size(); i++){
                denominator += std::exp(inputvec[i]);
            }
            for (int i = 0; i < inputvec.size(); i++){
                outputvec.push_back(std::exp(inputvec[i])/denominator);
            }
        }

        double mse(std::vector<double> predictions, std::vector<double> labels){
            double error;
            for (int i = 0; i < predictions.size(); i++){
                error += ((labels[i]-predictions[i])*(labels[i]-predictions[i]));
            }
            return error;
            return 0;
        }

    //loss functions and stuff above
    //perceptron below

        std::vector<double> relu(std::vector<double> inputvec){
            std::vector<double> outputvec;
            for (int i = 0; i < inputvec.size(); i++){
                if (inputvec[i] < 0){
                    outputvec.push_back(0);
                }
                else{
                    outputvec.push_back(inputvec[i]);
                }
            }
            return outputvec;
        }

        double nofunction(double input){
            return input;
        }


        Perceptron::Perceptron(int inputsize, double learningrate, std::function<double(double)> activation){
            for (int i=0; i < inputsize; i++){
                weights.push_back(0.0);
            }
            this->learningrate = learningrate;
            this->activation = activation;
            bias = 0.0;
        }

        double Perceptron::predict(const std::vector<double>& inputs){
            double sum = bias;
            for (int i = 0; i < inputs.size(); i++){
                sum += (weights[i] * inputs[i]);
            }
            return nofunction(sum);
        }

        void Perceptron::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<double>& labels, int& epochs){
            for (int k = 0; k < epochs; k++){
                for (int i = 0; i < trainingdata.size(); i++){
                    double prediction = 0;
                    double sum = bias;
                    for (int j = 0; j < trainingdata[i].size(); j++){
                        sum += (weights[j] * trainingdata[i][j]);
                    }
                    prediction = nofunction(sum);

                    if (prediction != labels[i]){
                        for (int j = 0; j < weights.size(); j++){
                            weights[j] = weights[j] + (learningrate * labels[i] * trainingdata[i][j]);
                        }
                        bias = bias + (learningrate * labels[i]);
                    }
                }
            }
        }

//perceptron above
//mlp below

        Mlp::Mlp(const std::vector<int>& layersizes, double learningrate){
            this->learningrate = learningrate;

            this->layers.resize(layersizes.size()-1);

            for (int i = 1; i < (layersizes.size()-1); i++){
                this->layers[i-1].emplace_back(Perceptron(1, learningrate, nofunction));
            }
            for (int i = 1; i < layersizes[layersizes.size()-1]; i++){
                this->layers[layersizes.size()-1].emplace_back(Perceptron(1, learningrate, nofunction));
            }
        }

        std::vector<double> Mlp::predict(const std::vector<double>& inputs){
            std::vector<double> inputvec;
            std::vector<double> outputvec;
            for (int i = 0; i < layers[0].size(); i++){ //first pass, passes inputs into neurons initially
                inputvec.push_back(layers[0][i].predict(inputs));
            }
            for (int i = 1; i < layers.size(); i++){
                outputvec.clear();
                for (int j = 0; j < layers[i].size(); j++){
                    outputvec.push_back(layers[i][j].predict(inputvec));
                }
                inputvec = std::move(outputvec);
            }
            outputvec.clear();
            outputvec = std::move(relu(inputvec));
            return outputvec;
            
        }

        void Mlp::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<std::vector<double>>& labels, const int epochs){
            std::vector<double> labels;
            double error = 0;
            for (int i = 0; i < epochs; i++){
                for (int j = 0; j < labels.size(); j++){
                    error = mse(predict(trainingdata[j]), labels[j]);
                }
            }
        }



int main(){
    return 0;
}