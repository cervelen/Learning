#ifndef MLP_H
#define MLP_H

#include <vector>

class Perceptron{
    private:
        std::vector<double> weights;
        double bias;
        double learningrate;

    public:
        Perceptron(int inputsize, double learningrate);

        int predict(const std::vector<double>& inputs);
        void train(const std::vector<std::vector<double>>& trainingdata, const std::vector<int>& labels, int& epochs);

};

class Mlp{
    private:
        std::vector<std::vector<Perceptron>> layers;
        double learningrate;

    public:
        Mlp(const std::vector<int>& layersizes, double learningrate);

        std::vector<double> predict(const std::vector<double>& inputs);

        void train(const std::vector<std::vector<double>>& trainingdata, const std::vector<int>& labels, const int epochs);

};




#endif // MLP_H