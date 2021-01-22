
public class DataNet {
    public int InputNeurons = 100;
    public int HiddenNeurons = 50;
    public int Output_Shapes = 3;
    public int Low = -1;
    public int Hi = 1;
    public int X = 5;
    public int Y = 5;
    public int TrainPatt = 50;
    public int TestPatt = 10;

    public class InArr{
        public int[] inarr;
        public InArr(){
            inarr = new int[InputNeurons];
        }

    }

    public InArr[] Input;
    public int Output[];
    public int units;

    public DataNet(){
        this.Input = null;
        this.Output = new int[Output_Shapes];
        this.units = 0;

    }

    public boolean SetInputOutput(String[][] In,char[] Out, int number) {
        int n, i, j;

        if (units != number) {
            units = number;
            this.Input = new InArr[number];
        }

        for(n=0; n < units; n++){ //Set input vectors.
            for(i=0; i < Y; i++){
                for(j=0; j < (X-1); j++)
                    Input[n].inarr[i*(X)+j] = (In[n][i].charAt(j) == '*') ? Hi : Low;
            }
        }

        //Set corresponding to input expected output.
        for(i=0; i < units; i++){
            switch (Out[i]){
                case '*':
                    Output[i] = Shapes.rectangle.ordinal();

                case '+':
                    Output[i] = Shapes.triangular.ordinal();

                case '_':
                    Output[i] = Shapes.trapeze.ordinal();

            }
        }
        return true;
    }

}
