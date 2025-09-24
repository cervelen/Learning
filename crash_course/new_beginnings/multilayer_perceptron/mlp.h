#ifndef MLP_H
#define MLP_H

#include <vector>

std::vector<double> softmax(std::vector<double> inputvec);
double mse(std::vector<double> predictions, std::vector<double> labels);
std::vector<double> relu(std::vector<double> inputvec){

}

class Perceptron{
    private:
        std::vector<double> weights;
        double bias;
        double learningrate;
        std::function<double(double)> activation;

    public:
        Perceptron(int inputsize, double learningrate, std::function<double(double)> activation);

        double predict(const std::vector<double>& inputs);
        void Perceptron::train(const std::vector<std::vector<double>>& trainingdata, const std::vector<double>& labels, int& epochs);

};

class Mlp{
    private:
        std::vector<std::vector<Perceptron>> layers;
        double learningrate;

    public:
        Mlp(const std::vector<int>& layersizes, double learningrate);

        std::vector<double> predict(const std::vector<double>& inputs);

        void train(const std::vector<std::vector<double>>& trainingdata, const std::vector<std::vector<double>>& labels, const int epochs);

};




#endif // MLP_H