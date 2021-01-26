
import static java.lang.Math.tanh;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
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
		this.HiddenLayer=new double [51]; //50 hidden neurons.
		this.WeigthsOut=new double[51];// weights of the output neuron.
		this.WeigthsHidd = new double[51][101];//weights of the hidden neurons.
		this.OutputLayer=100;// one output neuron.
		this.Initialize();

	}

	// *---------------------Private methods----------------------*

	private double RandomEqualReal(double Low, double High) {
		return ((double) Math.random()) * (High - Low) + Low;
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

			HiddenLayer[i] = (double)tanh (Sum);
		}
		//Calculate output for output layer.
		Sum = 0.0;

		for(int n=0; n < HiddenNeurons; n++)
			Sum += WeigthsOut[n] * HiddenLayer[n];

		//System.out.println("sum "+tanh (Sum));
//		if ((float)tanh (Sum) < (-1*Threshold) )
//			OutputLayer = 0;
//
//		else if ( (float)tanh (Sum) > Threshold )
//			OutputLayer = 1;
//		else						                     //We can not decide.
//			OutputLayer = 2;
		//Make decision about output neuron.
		if (tanh (Sum)>=0.0 && tanh (Sum)<third ){
			OutputLayer = 0; // for rectangle
//			System.out.println("malben");
		}

		else if ( tanh (Sum)>=third && tanh (Sum)<doubleThird){
			OutputLayer = 1; // for triangular
//			System.out.println("mesolash");
		}

		else{
			OutputLayer = 2; //for trapeze
//			System.out.println("trapeze");
		}

	}


	//NetError = true if it was error.
	private void ItIsError(int Target){
		if(((double)Target - OutputLayer) != 0)
			this.NetError = true;
		else
			this.NetError = false;
	}


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
			outStream.write(("\n"+Success + "% success").getBytes());
		}while(Success < 90 && loop <= 20000);
		if(loop > 20000)
		{
			outStream.write(("\n"+"Training of network failure !").getBytes());
			return false;
		}
		return true;
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

	public int ReturnOutput() {
		return this.OutputLayer;
	}

	public double LearningRate() {
		return this.nu;
	}

	public static void main(String[] args) throws IOException {
		DataNet data_obj = new DataNet();
		BackPropagationNet back_prop_obj =new BackPropagationNet();
		boolean flag;
		File tempFile = new File("/Neural-network/test.txt");
		boolean exists = tempFile.exists();
		File path = new File("results.txt");
		if(path.exists()){ path.delete();}  // delete if exist and create a new one
		OutputStream outStream = new FileOutputStream(path);
		data test=new data();
		String output = "_+*+*_*+_";
		test.setStudy_group(3);
		outStream.write(("\n"+"Here we go").getBytes());
		if(! data_obj.SetInputOutput(test.getStudy_group(),output, 9))
			return;
		while( (flag =! back_prop_obj.TrainNet( data_obj, outStream )))
		{
			back_prop_obj.Initialize();
		}
		output = "+_*";
		if(!data_obj.SetInputOutput(test.getTest_group(),output, 3))
			return;
		back_prop_obj.TestNet(data_obj, outStream );
		outStream.close();
	}
}
