#ifndef PERCEPTRON_H
#define PERCEPTRON_H

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


#endif // PERCEPTRON_H