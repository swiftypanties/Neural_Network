import java.util.Comparator;

public class test implements Comparator<Double> {
    public static void main(String[] args){
        double a=5.6;
        double b=5.3;
        int g=(int)(Math.random()*10+1);
        String s=""+200;
        while(s.length()<8){s="0"+s;}
        System.out.println("s "+s);

        System.out.println("g "+g);
        System.out.println((int)a+1);
        test n= new test();
        int c=n.answer(b,a);
        System.out.println("c "+c);
        int [] r= new int[4];
        System.out.println(r.length);

    }
    public test(){

    }
    public int answer(double a, double b){
        return compare(a,b);
    }
    @Override
    public int compare(Double o1, Double o2) {
        if(o1<o2) return -1;
        if(o1>o2) return 1;
        return 0;
    }
}
