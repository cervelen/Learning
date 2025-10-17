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
            return error/labels.size();
            return 0;
        }

        double reluderivative(double &z){
            if (z > 0){
                return 1;
            }
            else{
                return 0;
            }
        }

    //loss functions and stuff above
    //perceptron below
        double singlerelu(double input){
            if (input < 0){
                return 0;
            }
            else{
                return input;
            }
        }
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
                this->layers[i-1].emplace_back(Perceptron(layersizes[0], learningrate, nofunction));
            }
            for (int i = 1; i < layersizes[layersizes.size()-1]; i++){
                this->layers[layersizes.size()-1].emplace_back(Perceptron(layers[i-1].size(), learningrate, nofunction));
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
            double prediction = 0;
            double error;
            std::vector<std::vector<double>> preactivationvalues;
            std::vector<std::vector<double>> outputvalues;
            for (int i = 0; i < epochs; i++){
                for (int j = 0; j < labels.size(); j++){
                    prediction = predict(trainingdata[j])[0];   //make prediction
                    for (int k = 0; k < layers[0].size(); k++){
                        preactivationvalues[0].push_back(layers[0][k].predict(trainingdata[j]));
                        outputvalues[0].push_back(singlerelu(preactivationvalues[0][k]));
                    }
                    for (int k = 1; k < layers.size(); k++){
                        for (int l = 0; l < layers[k].size(); l++){
                            preactivationvalues[k].push_back(layers[k][l].predict(outputvalues[k-1]));
                            outputvalues[k].push_back(preactivationvalues[k][l]);
                        }
                    }
                    

                    for (int k = layers[layers.size()-1].size()-1; k >= 0; k--){ //iterating backwards through the last layer's neurons, because it is funny.
                        error = (outputvalues[layers.size()-1][k] - labels[j][k])*reluderivative(preactivationvalues[layers.size()-1][k]);
                        for (int l = layers[layers.size()-1][k].weights.size()-1; l >= 0; l--){ //iterating, again backwards, through each neuron's weights, because it is funny.
                            layers[layers.size()-1][k].weights[l] = layers[layers.size()-1][k].weights[l] - learningrate * error * outputvalues[layers.size()][k];
                        }
                        layers[layers.size()][k].bias = layers[layers.size()][k].bias - learningrate * error;
                    }
                    for (int k = layers.size()-2; k >= 0; k--){
                        for (int l = layers[k].size()-1; l >= 0; l--){ //iterating backwards through each layers neurons, because it is funny.
                            error = 0;
                            for (int m = layers[k+1].size()-1; m >= 0; m--){
                                error += layers[k+1][m].weights[k]*errors[k+1][m];
                            }
                            error = reluderivative(preactivationvalues[k][l])*error;
                        }
                    }
                }
            }
        }



int main(){
    return 0;
}