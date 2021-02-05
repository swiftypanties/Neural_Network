import static java.lang.Math.exp;
import static java.lang.Math.tanh;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.*;

public class BackPropagationNetTwoLayer  {

    //*-----------Final variables-------------
    public final int Low = -1;
    public final int Hi = 1;
    public final int InputNeurons = 100;
    public final int  HiddenNeurons_firstLayer = 50;
    public final int  HiddenNeurons_secondLayer = 25;

    //*-----------Local variables-------------
    private double nu; //The learning rate parameter.
    private boolean NetError;
    // value and array of the output layer.
    private double OutputLayer;
    private double[] WeigthsOut;
    // first hidden layer
    private double[] first_HiddenLayer;
    private double[][] Weigths_first_Hidd;
    // second hidden layer
    private double[] second_HiddenLayer;
    private double[][] Weigths_second_Hidd;
    // input layer
    private int[] InputLayer;

    //*-----------Constructor-------------
    public BackPropagationNetTwoLayer() {
        this.nu=0.1;
        this.InputLayer=new int[101];// 100 input neurons.
        this.first_HiddenLayer =new double [51];
        this.Weigths_first_Hidd = new double[51][101];
        this.second_HiddenLayer =new double [26];
        this.Weigths_second_Hidd = new double[26][51];
        this.WeigthsOut=new double[26];
        this.OutputLayer=100;// one output neuron.
        this.Initialize();
    }
    // *---------------------Private methods----------------------*

    private double RandomEqualReal(double Low, double High) {
        double ans= ((double) Math.random()) * (High - Low) + Low;
        return ans;
    }
    private double sigmoid(double Sum){
        double func=1.0/(1+exp(-Sum));
        return func;
    }

    //Calculate output for current input (without Bias).
    private void CalculateOutput(){
        double Sum;
        double third= 1.0/3.0;
        double doubleThird=2*third;

        //Calculate output for the first hidden layer.
        for(int i=0; i < HiddenNeurons_firstLayer; i++)
        {
            Sum = 0.0;
            for(int j=0; j < InputNeurons; j++)
            {
                Sum += Weigths_first_Hidd[i][j] * InputLayer[j];
            }
            first_HiddenLayer[i] = sigmoid(Sum);
        }
        //Calculate output for the second hidden layer.
        for(int i=0; i < HiddenNeurons_secondLayer; i++)
        {
            Sum = 0.0;
            for(int j=0; j < HiddenNeurons_firstLayer; j++)
            {
                Sum += Weigths_second_Hidd[i][j] * first_HiddenLayer[j];
            }
            second_HiddenLayer[i] = sigmoid(Sum);
        }
        //Calculate output for output layer.
        Sum = 0.0;

        for(int n=0; n < HiddenNeurons_secondLayer; n++)
            Sum += WeigthsOut[n] * second_HiddenLayer[n];

        if (sigmoid(Sum)>=0.0 && sigmoid(Sum)<third ){
            this.OutputLayer = 0; // for triangular(mesolash)
        }

        else if (sigmoid(Sum)>=third && sigmoid(Sum)<doubleThird){
            this.OutputLayer = 1; // for trapeze
        }

        else{
            this.OutputLayer = 2; //for rectangle(malben)
        }
    }


    //NetError = true if it was error.
    private void ItIsError(int Target){
        if(((double)Target - this.OutputLayer) != 0.0)
            this.NetError = true;
        else
            this.NetError = false;
    }
    private double divSigmoid(double OutputLayer){
        return sigmoid(OutputLayer)*(1-sigmoid(OutputLayer));
    }
    private void AdjustWeights(int Target){
        int i, j;
        double sum=0.0;
        double[] hidd_deltas_firstLayer= new double[HiddenNeurons_firstLayer];
        double[] hidd_deltas_secondLayer= new double[HiddenNeurons_secondLayer];
        double out_delta;

        //Calcilate deltas for the output layer
        out_delta = divSigmoid(OutputLayer) * (Target - OutputLayer);

        //Calcilate deltas for the second hidden layer.
        for(i=0; i < HiddenNeurons_secondLayer; i++)
            hidd_deltas_secondLayer[i] = divSigmoid(second_HiddenLayer[i]) * out_delta * WeigthsOut[i];

        //Change weigths of output layer.
        for(i=0; i < HiddenNeurons_secondLayer; i++)
            WeigthsOut[i] = WeigthsOut[i]+(nu * out_delta * second_HiddenLayer[i]);

        //Calcilate deltas for the first hidden layer.
        for(i=0; i < HiddenNeurons_firstLayer; i++){
            sum=0.0;
            for(j=0;j<HiddenNeurons_secondLayer;j++){
                sum+=hidd_deltas_secondLayer[j] * Weigths_second_Hidd[j][i];
            }
            hidd_deltas_firstLayer[i]=divSigmoid(first_HiddenLayer[i]) * sum;
        }

        //Change weigths of the second hidden layer.
        for(i=0; i < HiddenNeurons_secondLayer; i++)
        {
            for(j=0; j < HiddenNeurons_firstLayer+1; j++)
                Weigths_second_Hidd[i][j] = Weigths_second_Hidd[i][j] + (nu * hidd_deltas_secondLayer[i] * first_HiddenLayer[j]);
        }
        //Change weigths of the first hidden layer.
        for(i=0; i < HiddenNeurons_firstLayer; i++)
        {
            for(j=0; j < InputNeurons+1; j++)
                Weigths_first_Hidd[i][j] = Weigths_first_Hidd[i][j] + (nu * hidd_deltas_firstLayer[i] * InputLayer[j]);
        }

    }


    //-------------Public methods--------------
    public void Initialize() {
        this.NetError=false;// no error at the beginning.
        // init the weight array of the output layer: 50 weights.
        for(int i=0;i<26;i++) {
            this.WeigthsOut[i]= RandomEqualReal(-1.0,1.0);
        }
        for(int i=0;i<26;i++) {
            for(int j=0;j<51;j++) {
                Weigths_second_Hidd[i][j] = RandomEqualReal(-1.0, 1.0);

            }
        }
        for(int i=0;i<51;i++) {
            for(int j=0;j<101;j++) {
                Weigths_first_Hidd[i][j] = RandomEqualReal(-1.0, 1.0);

            }
        }

    }

    public boolean TrainNet(DataNet _data, OutputStream outStream) throws IOException {
        int Error, j, Success, loop = 0;
        do
        {
            Error = 0;
            loop ++;
            outStream.write(("\n"+"Training loop num : "+ loop).getBytes());

            //Train network (do one cycle).
            for(int i=0; i < _data.units; i++)
            {
                //Set current input.
                for(j=0; j < InputNeurons; j++)
                    InputLayer[j] = _data.Input[i][j];
                CalculateOutput();
                ItIsError(_data.Output[i]);
                if(this.NetError)
                {
                    Error ++;
                    AdjustWeights(_data.Output[i]);
                }
            }
            Success = ((_data.units - Error)*100) / _data.units;
            outStream.write(("\n"+Success + "% success"+"\n").getBytes());
        }while(Success < 90 && loop <= 20000);
        if(loop > 20000)
        {
            outStream.write(("\n"+"Training of network failure !").getBytes());
            return false;
        }
        else return true;
    }

    public int TestNet(DataNet _data, OutputStream outStream) throws IOException {
        int Error = 0, j, Success,i;
        int averageSuccess = 0;
        //Train network (do one cycle).
        for(i=0; i < _data.units; i++)
        {
            //Set current input.
            for(j=0; j < InputNeurons; j++)
                InputLayer[j] = _data.Input[i][j];

            CalculateOutput();
            ItIsError(_data.Output[i]);

            //Error = sum of errors in this one cycle of test.
            if(this.NetError)
                Error ++;
            Success = ((_data.units - Error)*100) / _data.units;
            averageSuccess = averageSuccess + Success;
            outStream.write(("\n"+"The Average test success rate is: " +(averageSuccess/i) + "%\n").getBytes());
        }
        Success = ((_data.units - Error)*100) / _data.units;
        outStream.write(("\n"+(averageSuccess/i) + "% success\n").getBytes());
        return Success;
    }

    public double ReturnOutput() {
        return this.OutputLayer;
    }

    public double LearningRate() {
        return this.nu;
    }

    public static void main(String[] args) throws IOException {

        BackPropagationNetTwoLayer back_prop_obj = new BackPropagationNetTwoLayer();
        boolean flag = true;
        //-------------------------- 4 sorted
        File path1 = new File("4 sorted two layers.txt");
        if (path1.exists()) {
            path1.delete();
        }  // delete if exist and create a new one
        OutputStream outStreamS4 = new FileOutputStream(path1);
        data test_sort4 = new data();
        DataNet data_obj = new DataNet();
        test_sort4.setStudy_group_sorted(4);
        back_prop_obj.Initialize();
        outStreamS4.write(("Start Train 4 sorted study groups- Two layers").getBytes());
        if (!data_obj.SetInputOutput(test_sort4.getStudy_group(), test_sort4.getOutput_result(), 12))
            return;

        while ((flag != back_prop_obj.TrainNet(data_obj, outStreamS4))) {
            back_prop_obj.Initialize();
        }
        data_obj = new DataNet();
        if (!data_obj.SetInputOutput(test_sort4.getTest_group1(), test_sort4.getOutput_test1(), 3))
            return;
        outStreamS4.write(("\nStart test 4 sorted study groups with 1 group- Two layers").getBytes());
        back_prop_obj.TestNet(data_obj, outStreamS4);
        outStreamS4.close();
        //-------------------------- 9 sorted
        back_prop_obj = new BackPropagationNetTwoLayer();
        back_prop_obj.Initialize();
        File path2 = new File("9 sorted two layers.txt");
        if (path2.exists()) {
            path2.delete();
        }  // delete if exist and create a new one
        OutputStream outStreamS9 = new FileOutputStream(path2);
        data test_sort9 = new data();
        data_obj = new DataNet();
        test_sort9.setStudy_group_sorted(9);
        outStreamS9.write(("\nStart Train 9 sorted study groups- Two layers").getBytes());
        if (!data_obj.SetInputOutput(test_sort9.getStudy_group(), test_sort9.getOutput_result(), 27))
            return;
        while ((flag != back_prop_obj.TrainNet(data_obj, outStreamS9))) {
            back_prop_obj.Initialize();
        }

        data_obj = new DataNet();
        if (!data_obj.SetInputOutput(test_sort9.getTest_group1(), test_sort9.getOutput_test1(), 3))
            return;
        outStreamS9.write(("\nStart test 9 sorted study groups with 1 group- Two layers").getBytes());
        back_prop_obj.TestNet(data_obj, outStreamS9);
        outStreamS9.close();


		//-------------------------- 15 sorted
		back_prop_obj =new BackPropagationNetTwoLayer();
		back_prop_obj.Initialize();
		File path3 = new File("15 sorted two layers.txt");
		if(path3.exists()){ path3.delete();}  // delete if exist and create a new one
		OutputStream outStreamS15 = new FileOutputStream(path3);
		data test_sort15 = new data();
		data_obj = new DataNet();
		test_sort15.setStudy_group_sorted(15);
		outStreamS15.write(("\nStart Train 15 sorted study groups- Two layers").getBytes());
		if(! data_obj.SetInputOutput(test_sort15.getStudy_group(), test_sort15.getOutput_result(), 15*3))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamS15 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_sort15.getTest_group1(),test_sort15.getOutput_test1(),3))
			return;
        outStreamS15.write(("\nStart test 15 sorted study groups with 1 group- Two layers").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamS15 );
		outStreamS15.close();

		//-------------------------- 4 random
		back_prop_obj =new BackPropagationNetTwoLayer();
		back_prop_obj.Initialize();
		File path4 = new File("4 random two layers.txt");
		if(path4.exists()){ path4.delete();}  // delete if exist and create a new one
		OutputStream outStreamR4 = new FileOutputStream(path4);
		data test_random4 = new data();
		data_obj = new DataNet();
		test_random4.setStudy_group_random(4);
		outStreamR4.write(("\nStart Train 4 random study groups- Two layers").getBytes());
		if(! data_obj.SetInputOutput(test_random4.getStudy_group(), test_random4.getOutput_result(), 12))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR4 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random4.getTest_group1(),test_random4.getOutput_test1(),3))
			return;
        outStreamR4.write(("\nStart test 4 random study groups with 1 group- Two layers").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR4 );
		outStreamS4.close();



		//-------------------------- 9 random
		back_prop_obj =new BackPropagationNetTwoLayer();
		back_prop_obj.Initialize();
		File path5 = new File("9 random two layers.txt");
		if(path5.exists()){ path5.delete();}  // delete if exist and create a new one
		OutputStream outStreamR9 = new FileOutputStream(path5);
		data test_random9 = new data();
		data_obj = new DataNet();
		test_random9.setStudy_group_random(9);
		outStreamR9.write(("\nStart Train 9 random study groups- Two layers").getBytes());
		if(! data_obj.SetInputOutput(test_random9.getStudy_group(), test_random9.getOutput_result(), 27))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR9 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random9.getTest_group1(),test_random9.getOutput_test1(),3))
			return;
        outStreamR9.write(("\nStart test 9 random study groups with 1 group- Two layers").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR9 );
		outStreamR9.close();


		//-------------------------- 15 random
		back_prop_obj =new BackPropagationNetTwoLayer();
		back_prop_obj.Initialize();
		File path6 = new File("15 random two layers.txt");
		if(path6.exists()){ path6.delete();}  // delete if exist and create a new one
		OutputStream outStreamR15 = new FileOutputStream(path6);
		data test_random15 = new data();
		data_obj = new DataNet();
		test_random15.setStudy_group_random(15);
		outStreamR15.write(("\nStart Train 15 random study groups- Two layers").getBytes());
		if(! data_obj.SetInputOutput(test_random15.getStudy_group(), test_random15.getOutput_result(), 15*3))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR15 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random15.getTest_group1(),test_random15.getOutput_test1(),3))
			return;
        outStreamR15.write(("\nStart test 15 random study groups with 1 group- Two layers").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR15 );
		outStreamR15.close();

    }

}