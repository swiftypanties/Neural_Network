
import static java.lang.Math.tanh;

import java.io.*;
import java.util.*;

public class BackPropagationNet  {
	//*====== add
	public int Output_Shapes = 3;
	public int X = 5;
	public int Y = 5;
	public int TrainPatt = 50;
	public int TestPatt = 10;

	//*-----------Final variables-------------
	public final int Low = -1;
	public final int Hi = +1;
	public final int InputNeurons = 25;
	public final int  HiddenNeurons = 10;
	public final double sqr(double x) {return x*x;}

	//*-----------Local variables-------------
	private double nu; //The learning rate parameter.
	private double Threshold;
	private double OutputLayer;
	private boolean NetError;
	private double[] HiddenLayer; //output of the hidden layer.
	private double[] WeigthsOut; //weights of the hidden layer.
	private int[] InputLayer; //input layer-> 100 inputs.
	private double[][] WeigthsHidd; // matrix

	//*-----------Constructor-------------
	public BackPropagationNet() {
		this.nu=0.1;
		this.HiddenLayer=new double [51];
		this.WeigthsOut=new double[51];
		this.InputLayer=new int[101];
		this.WeigthsHidd = new double[51][101];
		this.Initialize();

	}

	// *---------------------Private methods----------------------*

	private double RandomEqualReal(double Low, double High) {
		return ((double) Math.random()) * (High - Low) + Low;
	}

	//Calculate output for current input (without Bias).
	private void CalculateOutput(){
		double Sum;

		//Calculate output for hidden layer.
		for(int i=0; i < HiddenNeurons; i++)
		{
			Sum = 0.0f;
			for(int j=0; j < InputNeurons; j++)
			{
				Sum += WeigthsHidd[i][j] * InputLayer[j];
			}

			HiddenLayer[i] = (double)tanh (Sum);
		}
		//Calculate output for output layer.
		Sum = 0.0f;

		for(int n=0; n < HiddenNeurons; n++)
			Sum += WeigthsOut[n] * HiddenLayer[n];

		//Make decision about output neuron.
		if (tanh (Sum) > Threshold )
			OutputLayer = 1.0f;
		else if ( tanh (Sum) < - Threshold )
			OutputLayer = -1.0f;
		else						                     //We can not decide.
			OutputLayer = (double)tanh (Sum);
	}


	//NetError = true if it was error.
	private void ItIsError(int Target){
		if(((double)Target - OutputLayer) != 0)
			NetError = true;
		else
			NetError = false;
	}


	//Without Bias.
	private void AdjustWeights(int Target){
		int i, j;
		double[] hidd_deltas= new double[HiddenNeurons];
		double out_delta;

		//Calcilate deltas for all layers.
		out_delta = (1 - sqr(OutputLayer)) * (Target - OutputLayer);

		for(i=0; i < HiddenNeurons; i++)
			hidd_deltas[i] = (1 - sqr(HiddenLayer[i])) * out_delta * WeigthsOut[i];

		//Change weigths.
		for(i=0; i < HiddenNeurons; i++)
			WeigthsOut[i] = WeigthsOut[i]+(nu * out_delta * HiddenLayer[i]);

		for(i=0; i < HiddenNeurons; i++)
		{
			for(j=0; j < InputNeurons+1; j++)
				WeigthsHidd[i][j] = WeigthsHidd[i][j] +
						(nu * hidd_deltas[i] * InputLayer[j]);
		}
	}


	//-------------Public methods--------------
	public void Initialize() {
		this.Threshold=0.8;
		this.NetError=false;
		for(int i=0;i<51;i++) {
			this.WeigthsOut[i]= RandomEqualReal(-1.0,-1.0);
		}
		for(int i=0;i<51;i++) {
			for(int j=0;j<101;j++) {
					WeigthsHidd[i][j] = RandomEqualReal(-1.0, -1.0);
			}
		}
	}

	public boolean TrainNet(DataNet _data) {
		int Error, j, loop = 0, Success;
		do
		{
			Error = 0;
			loop ++;
			System.out.println("Im Looping at - "+ loop);

			//Train network (do one cycle).
			for(int i=0; i < _data.units; i++)
			{
				//Set current input.
				for(j=0; j < InputNeurons; j++)
					InputLayer[j] = _data.Input[i].inarr[j];
				CalculateOutput();
				ItIsError(_data.Output[i]);
				//If it was error, change weigths (Error = sum of errors in
				//one cycle of train).
				if(NetError)
				{
					Error ++;
					AdjustWeights(_data.Output[i]);
				}
			}
			Success = ((_data.units - Error)*100) / _data.units;
			System.out.println(Success + "% success");
			if( Success < 90 ) {Threshold = RandomEqualReal(0.2f, 0.9f);}
		}while(Success < 90 && loop <= 20000);
		if(loop > 20000)
		{
			System.out.println("Training of network failure !");
			return false;
		}
		return true;
	}

	public int TestNet(DataNet _data) {
		int Error = 0, j, Success;
		//Train network (do one cycle).
		for(int i=0; i < _data.units; i++)
		{
			//Set current input.
			for(j=0; j < InputNeurons; j++)
				InputLayer[j] = _data.Input[i].inarr[j];

			CalculateOutput();
			ItIsError(_data.Output[i]);

			//Error = sum of errors in this one cycle of test.
			if(NetError)
				Error ++;
		}
		Success = ((_data.units - Error)*100) / _data.units;
		System.out.println(Success+" % success");
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
		DataNet data_obj = new DataNet();
		BackPropagationNet back_prop_obj =new BackPropagationNet();
		boolean flag;
		File path = new File("results.txt");
		if(path.exists()){ path.delete();}  // delete if exist and create a new one
		OutputStream outStream = new FileOutputStream(path);
		outStream.write(("We can write what we want here").getBytes()); // this is how we will write into the file
		outStream.close();




		//TRAINING NETWORK WITHOUT BIAS.
//		if(! data_obj.SetInputOutput(TrainingInput, TrainingOutput, TrainPatt))
//			return;
//
//		while ( ! (flag = back_prop_obj.TrainNet( data_obj )))
//		{
//			back_prop_obj.Initialize();
//			close(fd);
//			remove("result.txt");
//			fd = open("result.txt", O_CREAT | O_RDWR, 0777);
//
//			if( fd == -1 )
//			{
//				cout << "Error opening result file" << endl;
//				return;
//			}
//		}
//
//		//TEST NETWORK.
//
//		if(! data_obj.SetInputOutput(TestInput, TestOutput, TestPatt))
//			return;
//
//		back_prop_obj.TestNet( data_obj );
//		close(fd);
	}
}
