
public class DataNet {
    public int InputNeurons = 100;
    public int HiddenNeurons = 50;
    public int Low = -1;
    public int Hi = 1;
    public int X = 1;
    public int Y = 10;


    public int[][] Input;
    public int[] Output;
    public int units;

    public DataNet(){
        this.units = 0;
    }

    public boolean SetInputOutput(char [][][] In,char[] Out, int number) {
        int n, i, j;

        if (units != number) {
            this.Input = new int[number][InputNeurons];
            this.Output = new int[number];
            this.units = number;

        }

        for(n=0; n < units; n++){ //Set input vectors.
            for(i=0; i < 10; i++){
                for(j=0; j < 10; j++) {
                    if (In[n][i][j] == '*') {Input[n][i*(10) + j] = Hi;}
                    else {Input[n][i*(10) + j] = Low; }
                }
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
