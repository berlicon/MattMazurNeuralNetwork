using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MattMazurNeuralNetwork
{
    //Реализация нейросети из статьи Matt Mazur "A Step by Step Backpropagation Example"
    //https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    class Program
    {
        private static double[] layerInputNodes = new double[2];
        private static double[] layerAssociationsNodes = new double[2];
        private static double[] layerAssociationsWeights = new double[4];
        private static double[] layerAssociationsWeightDeltas = new double[4];
        private static double[] layerResultNodes = new double[2];
        private static double[] layerResultWeights = new double[4];
        private static double[] layerResultWeightPartialDeltas = new double[4];
        private static double[] layerResultWeightDeltas = new double[4];
        private static double[] biasSignals = new double[2];
        private static double[] biasWeights = new double[2];
        private static double[] layerResultNodesTarget = new double[2];

        private static double learningRate = 0.5;
        private static double error;

        static void Main(string[] args)
        {
            initWeights();
            train();
        }

        private static void train()
        {
            Console.WriteLine("Начало тренировки нейросети");
            for (int i = 0; i < 10000; i++)
            {
                Console.WriteLine("Итерация {0} из {1}", i + 1, 10000);
                calculateAssociationsLayer();
                calculateResultLayer();
                printTotalError();
                backPropagation();
            }
        }

        private static void initWeights()
        {
            layerInputNodes[0] = .05;
            layerInputNodes[1] = .10;

            layerAssociationsWeights[0] = .15;
            layerAssociationsWeights[1] = .20;
            layerAssociationsWeights[2] = .25;
            layerAssociationsWeights[3] = .30;

            layerResultWeights[0] = .40;
            layerResultWeights[1] = .45;
            layerResultWeights[2] = .50;
            layerResultWeights[3] = .55;

            biasSignals[0] = 1;
            biasSignals[1] = 1;
            biasWeights[0] = .35;
            biasWeights[1] = .60;

            layerResultNodesTarget[0] = .01;
            layerResultNodesTarget[1] = .99;
        }

        private static void printTotalError()
        {
            error = 0;
            for (int i = 0; i < 2; i++)
            {
                error += (0.5 * Math.Pow(layerResultNodesTarget[i] - layerResultNodes[i], 2));
            }
            Console.WriteLine("Ошибка нейросети: {0}", error.ToString("F10"));
        }

        private static double funActivation(double value)
        {
            return (1 / (1 + Math.Pow(Math.E, -value)));
        }

        private static void calculateAssociationsLayer()
        {
            layerAssociationsNodes[0] =
                biasSignals[0] * biasWeights[0]
                + layerInputNodes[0] * layerAssociationsWeights[0]
                + layerInputNodes[1] * layerAssociationsWeights[1];
            layerAssociationsNodes[0] = funActivation(layerAssociationsNodes[0]);

            layerAssociationsNodes[1] =
                biasSignals[0] * biasWeights[0]
                + layerInputNodes[0] * layerAssociationsWeights[2]
                + layerInputNodes[1] * layerAssociationsWeights[3];
            layerAssociationsNodes[1] = funActivation(layerAssociationsNodes[1]);
        }

        private static void calculateResultLayer()
        {
            layerResultNodes[0] =
                biasSignals[1] * biasWeights[1]
                + layerAssociationsNodes[0] * layerResultWeights[0]
                + layerAssociationsNodes[1] * layerResultWeights[1];
            layerResultNodes[0] = funActivation(layerResultNodes[0]);

            layerResultNodes[1] =
                biasSignals[1] * biasWeights[1]
                + layerAssociationsNodes[0] * layerResultWeights[2]
                + layerAssociationsNodes[1] * layerResultWeights[3];
            layerResultNodes[1] = funActivation(layerResultNodes[1]);
        }

        private static void backPropagation()
        {
            calculateLayerResultWeightDeltas();
            calculateLayerAssociationsWeightDeltas();
            updateLayerResultWeightDeltas();
            updateLayerAssociationsWeightDeltas();
        }

        private static void calculateLayerResultWeightDeltas()
        {
            layerResultWeightPartialDeltas[0] =
                (layerResultNodesTarget[0] - layerResultNodes[0])
                * layerResultNodes[0]
                * (1 - layerResultNodes[0]);
            layerResultWeightDeltas[0] =
                layerResultWeightPartialDeltas[0]
                * layerAssociationsNodes[0];

            layerResultWeightPartialDeltas[1] =
                (layerResultNodesTarget[0] - layerResultNodes[0])
                * layerResultNodes[0]
                * (1 - layerResultNodes[0]);
            layerResultWeightDeltas[1] =
                layerResultWeightPartialDeltas[1]
                * layerAssociationsNodes[1];

            layerResultWeightPartialDeltas[2] =
                (layerResultNodesTarget[1] - layerResultNodes[1])
                * layerResultNodes[1]
                * (1 - layerResultNodes[1]);
            layerResultWeightDeltas[2] =
                layerResultWeightPartialDeltas[2]
                * layerAssociationsNodes[0];

            layerResultWeightPartialDeltas[3] =
                (layerResultNodesTarget[1] - layerResultNodes[1])
                * layerResultNodes[1]
                * (1 - layerResultNodes[1]);
            layerResultWeightDeltas[3] =
                layerResultWeightPartialDeltas[3]
                * layerAssociationsNodes[1];
        }

        private static void calculateLayerAssociationsWeightDeltas()
        {
            layerAssociationsWeightDeltas[0] =
                (layerResultWeightPartialDeltas[0] * layerResultWeights[0]
                + layerResultWeightPartialDeltas[2] * layerResultWeights[2])
                * layerAssociationsNodes[0]
                * (1 - layerAssociationsNodes[0])
                * layerInputNodes[0];
            layerAssociationsWeightDeltas[1] =
                (layerResultWeightPartialDeltas[0] * layerResultWeights[0]
                + layerResultWeightPartialDeltas[2] * layerResultWeights[2])
                * layerAssociationsNodes[0]
                * (1 - layerAssociationsNodes[0])
                * layerInputNodes[1];
            layerAssociationsWeightDeltas[2] =
                (layerResultWeightPartialDeltas[1] * layerResultWeights[1]
                + layerResultWeightPartialDeltas[3] * layerResultWeights[3])
                * layerAssociationsNodes[1]
                * (1 - layerAssociationsNodes[1])
                * layerInputNodes[0];
            layerAssociationsWeightDeltas[3] =
                (layerResultWeightPartialDeltas[1] * layerResultWeights[1]
                + layerResultWeightPartialDeltas[3] * layerResultWeights[3])
                * layerAssociationsNodes[1]
                * (1 - layerAssociationsNodes[1])
                * layerInputNodes[1];
        }

        private static void updateLayerResultWeightDeltas()
        {
            for (int i = 0; i < 4; i++)
            {
                layerResultWeights[i] += (learningRate * layerResultWeightDeltas[i]);
            }
        }

        private static void updateLayerAssociationsWeightDeltas()
        {
            for (int i = 0; i < 4; i++)
            {
                layerAssociationsWeights[i] += (learningRate * layerAssociationsWeightDeltas[i]);
            }
        }
    }
}
