
import static java.lang.Math.exp;
import static java.lang.Math.tanh;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.*;

public class BackPropagationNet  {
	//*====== add
	public int Output_Shapes = 3;
	public int TrainPatt = 50;
	public int TestPatt = 10;

	//*-----------Final variables-------------
	public final int Low = -1;
	public final int Hi = 1;
	public final int InputNeurons = 100;
	public final int  HiddenNeurons = 50;
	public final double sqr(double x) {return x*x;}

	//*-----------Local variables-------------
	private double nu; //The learning rate parameter.
	private double Threshold;
	private int OutputLayer;
	private boolean NetError;
	private double[] HiddenLayer;
//	private double[] HiddenLayer2;
	private double[] WeigthsOut;
	private int[] InputLayer;
	private double[][] WeigthsHidd;
//	private double[][] WeigthsHidd2;

	//*-----------Constructor-------------
	public BackPropagationNet() {
		this.nu=0.1;
		this.InputLayer=new int[101];// 100 input neurons.
		this.HiddenLayer =new double [51];
//		this.HiddenLayer2 =new double [51]; //50 hidden neurons.
		this.WeigthsOut=new double[51];// weights of the output neuron.
		this.WeigthsHidd = new double[51][101];//weights of the hidden neurons.
		this.OutputLayer=100;//
		this.Initialize();// one output neuron.

	}

	// *---------------------Private methods----------------------*

	private double RandomEqualReal(double Low, double High) {
		double ans= ((double) Math.random()) * (High - Low) + Low;
		System.out.println("weight "+ans);
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

		//Calculate output for hidden layer.
		for(int i=0; i < HiddenNeurons; i++)
		{
			Sum = 0.0;
			for(int j=0; j < InputNeurons; j++)
			{
				Sum += WeigthsHidd[i][j] * InputLayer[j];
			}

			HiddenLayer[i] = sigmoid(Sum);
		}
		//Calculate output for output layer.
		Sum = 0.0;

		for(int n=0; n < HiddenNeurons; n++)
			Sum += WeigthsOut[n] * HiddenLayer[n];

		//System.out.println("sum "+tanh (Sum));
		//Make decision about output neuron.
		System.out.println("sum "+Sum);
		System.out.println("sigmoid(Sum) "+sigmoid(Sum));

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
		double[] hidd_deltas= new double[HiddenNeurons];
		double out_delta;

		//Calcilate deltas for all layers.
		out_delta = divSigmoid(OutputLayer) * (Target - OutputLayer);

		for(i=0; i < HiddenNeurons; i++)
			hidd_deltas[i] = divSigmoid(OutputLayer) * out_delta * WeigthsOut[i];

		//Change weigths.
		for(i=0; i < HiddenNeurons; i++)
			WeigthsOut[i] = WeigthsOut[i]+(nu * out_delta * HiddenLayer[i]);

		for(i=0; i < HiddenNeurons; i++)
		{
			for(j=0; j < InputNeurons+1; j++)
				WeigthsHidd[i][j] = WeigthsHidd[i][j] + (nu * hidd_deltas[i] * InputLayer[j]);
		}
	}


	//-------------Public methods--------------
	public void Initialize() {
		this.Threshold=0.8;
		this.NetError=false;// no error at the beginning.
		// init the weight array of the output layer: 50 weights.
		for(int i=0;i<51;i++) {
			this.WeigthsOut[i]= RandomEqualReal(-1.0,1.0);
		}
		// init the weight array of the hidden layer: 100 weights for each hidden neuron.
		for(int i=0;i<51;i++) {
			for(int j=0;j<101;j++) {
				WeigthsHidd[i][j] = RandomEqualReal(-1.0, 1.0);

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
				System.out.println("output "+ReturnOutput());
				ItIsError(_data.Output[i]);
				//If it was error, change weigths (Error = sum of errors in
				//one cycle of train).
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
		int Error = 0, j, Success;
		//Train network (do one cycle).
		for(int i=0; i < _data.units; i++)
		{
			//Set current input.
			for(j=0; j < InputNeurons; j++)
				InputLayer[j] = _data.Input[i][j];

			CalculateOutput();
			ItIsError(_data.Output[i]);

			//Error = sum of errors in this one cycle of test.
			if(this.NetError)
				Error ++;
		}
		Success = ((_data.units - Error)*100) / _data.units;
		outStream.write(("\n"+Success + "% success").getBytes());
		return Success;
	}

	public double ReturnOutput() {
		return this.OutputLayer;
	}

	public double LearningRate() {
		return this.nu;
	}
	public double ThresholdValue() {
		return this.Threshold;
	}


	public static void main(String[] args) throws IOException {

		BackPropagationNet back_prop_obj = new BackPropagationNet();
		boolean flag = true;
		//-------------------------- 5 sorted
		File path1 = new File("5 sorted.txt");
		if (path1.exists()) {
			path1.delete();
		}  // delete if exist and create a new one
		OutputStream outStreamS5 = new FileOutputStream(path1);
		data test_sort5 = new data();
		DataNet data_obj = new DataNet();
		test_sort5.setStudy_group_sorted(5);
		data test_shira = new data();
		back_prop_obj.Initialize();
		outStreamS5.write(("Start Train 5 sorted study groups").getBytes());
		if(! data_obj.SetInputOutput(test_sort5.getStudy_group(), test_sort5.output_result, 15))
			return;
//		if (!data_obj.SetInputOutput(test_shira.getTestShira(), test_shira.getOutPutTestShira(), 15))
//			return;
		while ((flag != back_prop_obj.TrainNet(data_obj, outStreamS5))) {
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if (!data_obj.SetInputOutput(test_sort5.getTest_group(), test_sort5.getOutput_test(), 3))
			return;
		back_prop_obj.TestNet(data_obj, outStreamS5);
		outStreamS5.close();

//
//		//-------------------------- 10 sorted
//		back_prop_obj =new BackPropagationNet();
//		back_prop_obj.Initialize();
//		File path2 = new File("10 sorted.txt");
//		if(path2.exists()){ path2.delete();}  // delete if exist and create a new one
//		OutputStream outStreamS10 = new FileOutputStream(path2);
//		data test_sort10 = new data();
//		data_obj = new DataNet();
//		test_sort10.setStudy_group_sorted(10);
//		outStreamS10.write(("Start Train 10 sorted study groups").getBytes());
//		if(! data_obj.SetInputOutput(test_sort10.getStudy_group(), test_sort10.output_result, 30))
//			return;
//		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamS10 )))
//		{
//			back_prop_obj.Initialize();
//		}
//
//		data_obj = new DataNet();
//		if(!data_obj.SetInputOutput(test_sort10.getTest_group(),test_sort10.output_test,3))
//			return;
//		back_prop_obj.TestNet(data_obj, outStreamS10 );
//		outStreamS10.close();
//
//
//		//-------------------------- 19 sorted
//		back_prop_obj =new BackPropagationNet();
//		back_prop_obj.Initialize();
//		File path3 = new File("19 sorted.txt");
//		if(path3.exists()){ path3.delete();}  // delete if exist and create a new one
//		OutputStream outStreamS19 = new FileOutputStream(path3);
//		data test_sort19 = new data();
//		data_obj = new DataNet();
//		test_sort19.setStudy_group_sorted(19);
//		outStreamS19.write(("Start Train 19 sorted study groups").getBytes());
//		if(! data_obj.SetInputOutput(test_sort19.getStudy_group(), test_sort19.getOutput_result(), 19*3))
//			return;
//		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamS19 )))
//		{
//			back_prop_obj.Initialize();
//		}
//		data_obj = new DataNet();
//		if(!data_obj.SetInputOutput(test_sort19.getTest_group(),test_sort19.getOutput_test(),3))
//			return;
//		back_prop_obj.TestNet(data_obj, outStreamS19 );
//		outStreamS19.close();
//
//		//-------------------------- 5 random
//		back_prop_obj =new BackPropagationNet();
//		back_prop_obj.Initialize();
//		File path4 = new File("5 random.txt");
//		if(path4.exists()){ path4.delete();}  // delete if exist and create a new one
//		OutputStream outStreamR5 = new FileOutputStream(path4);
//		data test_random5 = new data();
//		data_obj = new DataNet();
//		test_random5.setStudy_group_random(5);
//		outStreamR5.write(("Start Train 5 random study groups").getBytes());
//		if(! data_obj.SetInputOutput(test_random5.getStudy_group(), test_random5.getOutput_result(), 15))
//			return;
//		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR5 )))
//		{
//			back_prop_obj.Initialize();
//		}
//		data_obj = new DataNet();
//		if(!data_obj.SetInputOutput(test_random5.getTest_group(),test_random5.getOutput_test(),3))
//			return;
//		back_prop_obj.TestNet(data_obj, outStreamR5 );
//		outStreamS5.close();
//
//
//
//		//-------------------------- 10 random
//		back_prop_obj =new BackPropagationNet();
//		back_prop_obj.Initialize();
//		File path5 = new File("10 random.txt");
//		if(path5.exists()){ path5.delete();}  // delete if exist and create a new one
//		OutputStream outStreamR10 = new FileOutputStream(path5);
//		data test_random10 = new data();
//		data_obj = new DataNet();
//		test_random10.setStudy_group_random(10);
//		outStreamR10.write(("Start Train 10 random study groups").getBytes());
//		if(! data_obj.SetInputOutput(test_random10.getStudy_group(), test_random10.getOutput_result(), 30))
//			return;
//		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR10 )))
//		{
//			back_prop_obj.Initialize();
//		}
//		data_obj = new DataNet();
//		if(!data_obj.SetInputOutput(test_random10.getTest_group(),test_random10.getOutput_test(),3))
//			return;
//		back_prop_obj.TestNet(data_obj, outStreamR10 );
//		outStreamR10.close();
//
//
//		//-------------------------- 19 random
//		back_prop_obj =new BackPropagationNet();
//		back_prop_obj.Initialize();
//		File path6 = new File("19 random.txt");
//		if(path6.exists()){ path6.delete();}  // delete if exist and create a new one
//		OutputStream outStreamR19 = new FileOutputStream(path6);
//		data test_random19 = new data();
//		data_obj = new DataNet();
//		test_random19.setStudy_group_random(19);
//		outStreamR19.write(("Start Train 19 random study groups").getBytes());
//		if(! data_obj.SetInputOutput(test_random19.getStudy_group(), test_random19.getOutput_result(), 19*3))
//			return;
//		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR19 )))
//		{
//			back_prop_obj.Initialize();
//		}
//		data_obj = new DataNet();
//		if(!data_obj.SetInputOutput(test_random19.getTest_group(),test_random19.getOutput_test(),3))
//			return;
//		back_prop_obj.TestNet(data_obj, outStreamR19 );
//		outStreamR19.close();
//	}
	}

}