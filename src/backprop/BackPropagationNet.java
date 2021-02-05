package backprop;

import static java.lang.Math.exp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

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
	private double[] WeigthsOut;
	private int[] InputLayer;
	private double[][] WeigthsHidd;


	//*-----------Constructor-------------
	public BackPropagationNet() {
		this.nu=0.1;
		this.InputLayer=new int[101];// 100 input neurons.
		this.HiddenLayer =new double [51];
		this.WeigthsOut=new double[51];// weights of the output neuron.
		this.WeigthsHidd = new double[51][101];//weights of the hidden neurons.
		this.OutputLayer=100;//
		this.Initialize();// one output neuron.

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

//		//Make decision about output neuron.

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
		if((Target - OutputLayer) != 0)
			this.NetError = true;
		else
			this.NetError = false;
	}
	private double divSigmoid(double OutputLayer){
		double ans= sigmoid(OutputLayer)*(1-sigmoid(OutputLayer));
		return ans;
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
			averageSuccess = averageSuccess+Success;
			outStream.write(("\n"+Success + "% success\n").getBytes());
		}
		Success = ((_data.units - Error)*100) / _data.units;
		outStream.write(("\n"+"The Average test success rate is: " +(averageSuccess/i) + "%\n").getBytes());
		return Success;
	}

	public int ReturnOutput() {
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
		//-------------------------- 4 sorted
		File path1 = new File("4 sorted one layer.txt");
		if (path1.exists()) {
			path1.delete();
		}  // delete if exist and create a new one
		OutputStream outStreamS4 = new FileOutputStream(path1);
		data test_sort4 = new data();
		DataNet data_obj = new DataNet();
		test_sort4.setStudy_group_sorted(4);
		back_prop_obj.Initialize();
		outStreamS4.write(("Start Train 4 sorted study groups- one layer").getBytes());
		if (!data_obj.SetInputOutput(test_sort4.getStudy_group(), test_sort4.getOutput_result(), 12))
			return;

		while ((flag != back_prop_obj.TrainNet(data_obj, outStreamS4))) {
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if (!data_obj.SetInputOutput(test_sort4.getTest_group1(), test_sort4.getOutput_test1(), 3))
			return;
		outStreamS4.write(("Start test 4 sorted study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamS4);
		outStreamS4.close();


		//-------------------------- 9 sorted
		back_prop_obj = new BackPropagationNet();
		back_prop_obj.Initialize();
		File path2 = new File("9 sorted one layer.txt");
		if (path2.exists()) {
			path2.delete();
		}  // delete if exist and create a new one
		OutputStream outStreamS9 = new FileOutputStream(path2);
		data test_sort9 = new data();
		data_obj = new DataNet();
		test_sort9.setStudy_group_sorted(9);
		outStreamS9.write(("Start Train 9 sorted study groups- one layer").getBytes());
		if (!data_obj.SetInputOutput(test_sort9.getStudy_group(), test_sort9.getOutput_result(), 27))
			return;
		while ((flag != back_prop_obj.TrainNet(data_obj, outStreamS9))) {
			back_prop_obj.Initialize();
		}

		data_obj = new DataNet();
		if (!data_obj.SetInputOutput(test_sort9.getTest_group1(), test_sort9.getOutput_test1(), 3))
			return;
		outStreamS9.write(("Start test 9 sorted study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamS9);
		outStreamS9.close();


		//-------------------------- 15 sorted
		back_prop_obj =new BackPropagationNet();
		back_prop_obj.Initialize();
		File path3 = new File("15 sorted one layer.txt");
		if(path3.exists()){ path3.delete();}  // delete if exist and create a new one
		OutputStream outStreamS15 = new FileOutputStream(path3);
		data test_sort15 = new data();
		data_obj = new DataNet();
		test_sort15.setStudy_group_sorted(15);
		outStreamS15.write(("Start Train 15 sorted study groups- one layer").getBytes());
		if(! data_obj.SetInputOutput(test_sort15.getStudy_group(), test_sort15.getOutput_result(), 15*3))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamS15 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_sort15.getTest_group1(),test_sort15.getOutput_test1(),3))
			return;
		outStreamS15.write(("Start test 15 sorted study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamS15 );
		outStreamS15.close();

		//-------------------------- 4 random
		back_prop_obj =new BackPropagationNet();
		back_prop_obj.Initialize();
		File path4 = new File("4 random one layer.txt");
		if(path4.exists()){ path4.delete();}  // delete if exist and create a new one
		OutputStream outStreamR4 = new FileOutputStream(path4);
		data test_random4 = new data();
		data_obj = new DataNet();
		test_random4.setStudy_group_random(4);
		outStreamR4.write(("Start Train 4 random study groups- one layer").getBytes());
		if(! data_obj.SetInputOutput(test_random4.getStudy_group(), test_random4.getOutput_result(), 12))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR4 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random4.getTest_group1(),test_random4.getOutput_test1(),3))
			return;
		outStreamR4.write(("Start test 4 random study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR4 );
		outStreamS4.close();



		//-------------------------- 9 random
		back_prop_obj =new BackPropagationNet();
		back_prop_obj.Initialize();
		File path5 = new File("9 random one layer.txt");
		if(path5.exists()){ path5.delete();}  // delete if exist and create a new one
		OutputStream outStreamR9 = new FileOutputStream(path5);
		data test_random9 = new data();
		data_obj = new DataNet();
		test_random9.setStudy_group_random(9);
		outStreamR9.write(("Start Train 9 random study groups- one layer").getBytes());
		if(! data_obj.SetInputOutput(test_random9.getStudy_group(), test_random9.getOutput_result(), 27))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR9 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random9.getTest_group1(),test_random9.getOutput_test1(),3))
			return;
		outStreamR9.write(("Start test 9 random study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR9 );
		outStreamR9.close();


		//-------------------------- 15 random
		back_prop_obj =new BackPropagationNet();
		back_prop_obj.Initialize();
		File path6 = new File("15 random one layer.txt");
		if(path6.exists()){ path6.delete();}  // delete if exist and create a new one
		OutputStream outStreamR15 = new FileOutputStream(path6);
		data test_random15 = new data();
		data_obj = new DataNet();
		test_random15.setStudy_group_random(15);
		outStreamR15.write(("Start Train 15 random study groups- one layer").getBytes());
		if(! data_obj.SetInputOutput(test_random15.getStudy_group(), test_random15.getOutput_result(), 15*3))
			return;
		while( (flag != back_prop_obj.TrainNet( data_obj, outStreamR15 )))
		{
			back_prop_obj.Initialize();
		}
		data_obj = new DataNet();
		if(!data_obj.SetInputOutput(test_random15.getTest_group1(),test_random15.getOutput_test1(),3))
			return;
		outStreamR15.write(("Start test 15 random study groups with 1 group- one layer").getBytes());
		back_prop_obj.TestNet(data_obj, outStreamR15 );
		outStreamR15.close();
	}

}