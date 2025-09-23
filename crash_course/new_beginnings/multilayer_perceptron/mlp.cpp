#include "mlp.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>


        double relu(double input){
            double output = 0;
            if (input > output){
                output = input;
            }
            return output;
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
            return activation(sum);
        }

        void Perceptron::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<double>& labels, int& epochs){
            for (int k = 0; k < epochs; k++){
                for (int i = 0; i < trainingdata.size(); i++){
                    double prediction = 0;
                    double sum = bias;
                    for (int j = 0; j < trainingdata[i].size(); j++){
                        sum += (weights[j] * trainingdata[i][j]);
                    }
                    prediction = activation(sum);

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
                this->layers[i-1].emplace_back(Perceptron(1, learningrate, relu));
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
            
        }

        void Mlp::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<int>& labels, const int epochs){

        }



int main(){
    return 0;
}