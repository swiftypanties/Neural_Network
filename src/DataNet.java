
public class DataNet {
    public int InputNeurons = 100;
    public int HiddenNeurons = 50;
    public int Output_Shapes = 3;
    public int Low = -1;
    public int Hi = 1;
    public int TrainPatt = 50;
    public int TestPatt = 10;


    public int[][] Input;
    public int Output[];
    public int units;

    public DataNet(){
        this.Input = null;
        this.Output = null ;
        this.units = 0;

    }

    public boolean SetInputOutput(String[][] In,String Out, int number) {
        int n, i, j;
        this.units = number;
        this.Input = new int[number][InputNeurons];

        for(n=0; n < units; n++){ //Set input vectors.
            for(i=0; i < 10; i++){
                for(j=0; j < 10; j++)
                    if(In[n][i].charAt(j) == '*'){
                        Input[n][i*(10)+j] = Hi;
                    }
                    else{
                        Input[n][i*(10)+j] = Low;
                    }
            }
        }
        this.Output  = new int[number];
        //Set corresponding to input expected output.
        for(i=0; i < Output.length; i++){
            switch (Out.charAt(i)){
                case '*': {
                    Output[i] = Shapes.triangular.ordinal();
                    break;
                }
                case '+': {
                    Output[i] = Shapes.trapeze.ordinal();
                    break;
                }

                case '_': {
                    Output[i] = Shapes.rectangle.ordinal();
                    break;
                }
            }
        }
        return true;
    }

}
