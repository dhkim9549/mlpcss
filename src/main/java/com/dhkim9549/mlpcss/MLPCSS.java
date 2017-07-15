package com.dhkim9549.mlpcss;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;
import java.io.*;

/**
 *  Building a CSS Model with MLP
 * @author Dong-Hyun Kim
 */
public class MLPCSS {

    static String hpId = "MLPCSS_h2_uSGD_mb16_ss16";

    //double learnigRate = Double.parseDouble(args[0]);
    static double learnigRate = 0.0025;

    // Number of sample size per iteration
    static long nSamples = 16;

    // Mini-batch size
    static int batchSize = 16;

    // Evaluation sample size
    static long nEvalSamples = 10000;

    static LineNumberReader in = null;
    static LineNumberReader in2 = null;
    static String trainingDataInputFileName = "/down/data/list.txt";
    static String testDataInputFileName = "/down/data/list.txt";

    public static void main(String[] args) throws Exception {

        System.out.println("************************************************");
        System.out.println("hpId = " + hpId);
        System.out.println("Number of hidden layers = 2");
        System.out.println("learnigRate = " + learnigRate);
        System.out.println("Updater = " + "SGD");
        System.out.println("mini-batch size (batchSize) = " + batchSize);
        System.out.println("Number of sample size per iteration (nSamples) = " + nSamples);
        System.out.println("i >= 0");
        System.out.println("************************************************");

        //MultiLayerNetwork model = getInitModel(learnigRate);
        MultiLayerNetwork model = readModelFromFile("/down/sin/css_model_MLPCSS_h2_uSGD_ge_mb16_ss16_ev100000_aSP_150000.zip");
        //MultiLayerNetwork model = readModelFromFile("/down/ttt_model_h2_uSGD_mb16_ss16_5210000.zip");

        NeuralNetConfiguration config = model.conf();
        System.out.println("config = " + config);

        // Training data input file reader
        in = new LineNumberReader(new FileReader(trainingDataInputFileName));

        // Training iteration
        long i = 0;

        while(true) {
            if(true) break;

            i++;

            if(i % 1000 == 0) {
                System.out.println("i = " + i);
            }
            if(i % 5000 == 0) {
                evaluateModel(model);
            }

            List<DataSet> listDs = getTrainingData();
            if(listDs.size() == 0) {
                break;
            }
            DataSetIterator trainIter = new ListDataSetIterator(listDs, batchSize);

            // Train the model
            model = train(model, trainIter);

            if (i % 50000 == 0) {
                writeModelToFile(model, "/down/css_model_" + hpId + "_" + i + ".zip");
            }
        }

        evaluateModelBatch(model);
    }

    public static MultiLayerNetwork getInitModel(double learningRate) throws Exception {

        int seed = 123;

        int numInputs = 3;
        int numOutputs = 2;
        int numHiddenNodes = 30;

        System.out.println("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.SGD)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        return model;
    }

    public static MultiLayerNetwork train(MultiLayerNetwork model, DataSetIterator trainIter) throws Exception {

        //model.setListeners(new ScoreIterationListener(1000));

        model.fit( trainIter );

        return model;
    }

    private static List<DataSet> getTrainingData() throws Exception {

        //System.out.println("Getting training data...");

        List<DataSet> listDs = new ArrayList<>();

        for (int i = 0; i < nSamples; i++) {

            String s = "";
            while(s.equals("") || s.startsWith("GUARNT_NO")) {
                s = in.readLine();
                if(s == null) {
                    in.close();
                    in = new LineNumberReader(new FileReader(trainingDataInputFileName));
                    s = in.readLine();
                }
            }

            DataSet ds = getDataSet(s);
            listDs.add(ds);
        }

        Collections.shuffle(listDs);

        //System.out.println("listDs.size() = " + listDs.size());
        //System.out.println("Getting training data complete.");

        return listDs;
    }

    public static String getToken(String s, int x) {
        return getToken(s, x, " \t\n\r\f");
    }

    public static String getToken(String s, int x, String delim) {

        s = s.replaceAll("\t", "\t ");

        StringTokenizer st = new StringTokenizer(s, delim);
        int counter = 0;
        String answer = null;
        while(st.hasMoreTokens()) {
            String token = st.nextToken();
            if(counter == x) {
                answer = token.trim();
            }
            counter++;
        }
        return answer;
    }

    private static DataSet getDataSet(String s) throws Exception {

        String guarnt_no = getToken(s, 0, "\t");
        String bad_yn = getToken(s, 19, "\t");
        long income = Long.parseLong(getToken(s, 15, "\t"));
        long debt = Long.parseLong(getToken(s, 16, "\t"));
        long cb_grd = Long.parseLong(getToken(s, 14, "\t"));

        double[] featureData = new double[3];
        double[] labelData = new double[2];

        featureData[0] = rescaleAmt(income);
        featureData[1] = rescaleAmt(debt);
        featureData[2] = (double)cb_grd / 10.0;
        if(bad_yn != null && bad_yn.equals("Y")) {
            labelData[0] = 1.0;
            labelData[1] = 0.0;
        } else {
            labelData[0] = 0.0;
            labelData[1] = 1.0;
        }

        INDArray feature = Nd4j.create(featureData, new int[]{1, 3});
        INDArray label = Nd4j.create(labelData, new int[]{1, 2});

        DataSet ds = new DataSet(feature, label);


/*        System.out.println("\nguarnt_no = " + guarnt_no);
        System.out.println(income + " " + debt + " " + cb_grd);
        System.out.println("ds = " + ds);
*/


        return ds;
    }

    public static void evaluateModel(MultiLayerNetwork model) {

        System.out.println("Evaluating...");

        System.out.println("income");
        long income = 0;
        for(int i = 0; i < 10; i++) {
            income = 10000000 * i;
            double[] featureData = new double[3];
            featureData[0] = rescaleAmt(income);
            featureData[1] = rescaleAmt(10000000);
            featureData[2] = 0.4;
            INDArray feature = Nd4j.create(featureData, new int[]{1, 3});
            INDArray output = model.output(feature);
            System.out.print("feature = " + feature);
            System.out.print("  output = " + output);
            double acc_rat = output.getDouble(0);
            System.out.println("  acc_rat = " + acc_rat);
        }

        System.out.println("debt");
        long debt = 0;
        for(int i = 0; i < 10; i++) {
            debt = 10000000 * i;
            double[] featureData = new double[3];
            featureData[0] = rescaleAmt(10000000);
            featureData[1] = rescaleAmt(debt);
            featureData[2] = 0.4;
            INDArray feature = Nd4j.create(featureData, new int[]{1, 3});
            INDArray output = model.output(feature);
            System.out.print("feature = " + feature);
            System.out.print("  output = " + output);
            double acc_rat = output.getDouble(0);
            System.out.println("  acc_rat = " + acc_rat);
        }

        System.out.println("cb_grd");
        for(int i = 1; i <= 10; i++) {
            double[] featureData = new double[3];
            featureData[0] = rescaleAmt(30000000);
            featureData[1] = rescaleAmt(40000000);
            featureData[2] = (double)i / 10.0;
            INDArray feature = Nd4j.create(featureData, new int[]{1, 3});
            INDArray output = model.output(feature);
            System.out.print("feature = " + feature);
            System.out.print("  output = " + output);
            double acc_rat = output.getDouble(0);
            System.out.println("  acc_rat = " + acc_rat);
        }
    }

    public static void evaluateModelBatch(MultiLayerNetwork model) throws Exception {

        System.out.println("Evaluating batch...");

        // Training data input file reader
        in2 = new LineNumberReader(new FileReader(testDataInputFileName));

        // Evaluation result output writer
        BufferedWriter out = new BufferedWriter(new FileWriter("/down/list_eval.txt"));

        int i = 0;

        String s = "";
        while((s = in2.readLine()) != null) {

            i++;
            if(i % 1000 == 0) {
                System.out.println("i = " + i);
            }

            String s2 = "";

            if(s.startsWith("GUARNT")) {

                s2 += "guarnt_no\t";
                s2 += "bad_yn\t";
                s2 += "income\t";
                s2 += "debt\t";
                s2 += "cb_grd\t";
                s2 += "acc_rat\t";

            } else {

                String guarnt_no = getToken(s, 0, "\t");
                String bad_yn = getToken(s, 19, "\t");
                long income = Long.parseLong(getToken(s, 15, "\t"));
                long debt = Long.parseLong(getToken(s, 16, "\t"));
                long cb_grd = Long.parseLong(getToken(s, 14, "\t"));

                double[] featureData = new double[3];
                featureData[0] = rescaleAmt(income);
                featureData[1] = rescaleAmt(debt);
                featureData[2] = (double) cb_grd / 10.0;
                INDArray feature = Nd4j.create(featureData, new int[]{1, 3});
                INDArray output = model.output(feature);
                System.out.print("feature = " + feature);
                System.out.print("  output = " + output);
                double acc_rat = output.getDouble(0);
                System.out.println("  acc_rat = " + acc_rat);

                s2 += guarnt_no + "\t";
                s2 += bad_yn + "\t";
                s2 += income + "\t";
                s2 += debt + "\t";
                s2 += cb_grd + "\t";
                s2 += acc_rat + "\t";
            }

            out.write(s + "\n");
            out.flush();
        }

        out.close();
    }

    public static double rescaleAmt(long x) {
        x = x + 10000000;
        long min = 10000000;
        long max = min + 100000000;
        double y = (Math.log(x) - Math.log(min)) / (Math.log(max) - Math.log(min));
        return y;
    }

    public static MultiLayerNetwork readModelFromFile(String fileName) throws Exception {

        System.out.println("Deserializing model...");

        // Load the model
        File locationToSave = new File(fileName);
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        System.out.println("Deserializing model complete.");

        return model;
    }

    public static void writeModelToFile(MultiLayerNetwork model, String fileName) throws Exception {

        System.out.println("Serializing model...");

        // Save the model
        File locationToSave = new File(fileName); // Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true; // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        System.out.println("Serializing model complete.");

    }
}
