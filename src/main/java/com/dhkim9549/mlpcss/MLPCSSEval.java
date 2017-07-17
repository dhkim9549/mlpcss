package com.dhkim9549.mlpcss;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * Evaluate a trained MLP CSS model
 */
public class MLPCSSEval {

    static LineNumberReader in = null;
    static String testDataInputFileName = "/down/data/list.txt";

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model = MLPCSS.readModelFromFile("/down/sin/css_model_MLPCSS_h2_uSGD_mb16_ss16_200000.zip");

        evaluateModelBatch(model);

    }

    public static void evaluateModelBatch(MultiLayerNetwork model) throws Exception {

        System.out.println("Evaluating batch...");

        // Training data input file reader
        in = new LineNumberReader(new FileReader(testDataInputFileName));

        // Evaluation result output writer
        BufferedWriter out = new BufferedWriter(new FileWriter("/down/list_eval.txt"));

        String header = "";
        header += "guarnt_no\t";
        header += "bad_yn\t";
        header += "cb_grd\t";
        header += "scor_grd\t";
        header += "acc_rat\t";
        out.write(header + "\n");
        out.flush();

        int i = 0;

        String s = "";
        while((s = in.readLine()) != null) {

            i++;
            if(i % 10000 == 0) {
                System.out.println("i = " + i);
            }

            if(s.startsWith("GUARNT")) {
                continue;
            }

            String acpt_updt_dy = MLPCSS.getToken(s, 11, "\t");
            if(acpt_updt_dy.compareTo("20150101") < 0) {
                continue;
            }

            String guarnt_no = MLPCSS.getToken(s, 0, "\t");
            String bad_yn = MLPCSS.getToken(s, 19, "\t");
            String scor_grd = MLPCSS.getToken(s, 13, "\t");
            long income = Long.parseLong(MLPCSS.getToken(s, 15, "\t"));
            long spos_annl_iamt = Long.parseLong(MLPCSS.getToken(s, 16, "\t"));
            long stot_debt_amt = Long.parseLong(MLPCSS.getToken(s, 17, "\t"));
            long spos_debt_amt = Long.parseLong(MLPCSS.getToken(s, 18, "\t"));
            long cb_grd = Long.parseLong(MLPCSS.getToken(s, 14, "\t"));

            double[] featureData = new double[5];
            double[] labelData = new double[2];

            featureData[0] = MLPCSS.rescaleAmt(income);
            featureData[1] = MLPCSS.rescaleAmt(spos_annl_iamt);
            featureData[2] = MLPCSS.rescaleAmt(stot_debt_amt);
            featureData[3] = MLPCSS.rescaleAmt(spos_debt_amt);
            featureData[4] = (double)cb_grd / 10.0;
            INDArray feature = Nd4j.create(featureData, new int[]{1, 5});
            INDArray output = model.output(feature);
            //System.out.print("feature = " + feature);
            //System.out.print("  output = " + output);
            double acc_rat = output.getDouble(0);
            //System.out.println("  acc_rat = " + acc_rat);

            String s2 = "";
            s2 += guarnt_no + "\t";
            s2 += bad_yn + "\t";
            s2 += cb_grd + "\t";
            s2 += scor_grd + "\t";
            s2 += acc_rat + "\t";

            out.write(s2 + "\n");
            out.flush();
        }

        out.close();
    }
}
