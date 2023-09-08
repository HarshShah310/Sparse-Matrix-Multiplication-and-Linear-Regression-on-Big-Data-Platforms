import java.io.*;
import java.util.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.*;
import org.apache.hadoop.mapreduce.lib.output.*;

// Class representing the custom Writable for Elem
class Elem implements Writable {
    short tag;      // 0 for M, 1 for N
    int index;      // one of the indexes, the other is used as a key
    double value;
    Elem() {
    }
    Elem(short tag, int index, double value) {
        this.tag = tag;
        this.index = index;
        this.value = value;
    }
    public short getTag() {
        return tag;
    }

    public int getIndex() {
        return index;
    }

    public double getValue() {
        return value;
    }
    @Override
    public void readFields(DataInput in) throws IOException {
        tag = in.readShort();
        index = in.readInt();
        value = in.readDouble();
    }
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeShort(tag);
        out.writeInt(index);
        out.writeDouble(value);
    }
}

// Class representing the custom WritableComparable for Pair
class Pair implements WritableComparable<Pair> {
    int i;
    int j;
    Pair() {
    }
    Pair(int i, int j) {
        this.i = i;
        this.j = j;
    }
    @Override
    public void readFields(DataInput in) throws IOException {
        i = in.readInt();
        j = in.readInt();
    }
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(i);
        out.writeInt(j);
    }
    @Override
    public String toString() {
        return i + "," + j;
    }
    @Override
    public int compareTo(Pair other) {
        if (i != other.i) {
            return Integer.compare(i, other.i);
        } 
        else {
            return Integer.compare(j, other.j);
        }
    }
}

public class Multiply {
    // Mapper for matrix M
    public static class MatrixMMapper extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] m_line = value.toString().split(",");
            int i = Integer.parseInt(m_line[0]);
            int j = Integer.parseInt(m_line[1]);
            double v = Double.parseDouble(m_line[2]);
            context.write(new IntWritable(j), new Elem((short) 0, i, v));
        }
    }
    // Mapper for matrix N
    public static class MatrixNMapper extends Mapper<Object, Text, IntWritable, Elem> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] n_line = value.toString().split(",");
            int i = Integer.parseInt(n_line[0]);
            int j = Integer.parseInt(n_line[1]);
            double v = Double.parseDouble(n_line[2]);
            context.write(new IntWritable(i), new Elem((short) 1, j, v));
        }
    }
    // Reducer for multiplication
    public static class MatrixReducer extends Reducer<IntWritable, Elem, Pair, DoubleWritable> {
        @Override
        public void reduce(IntWritable key, Iterable<Elem> values, Context context) throws IOException, InterruptedException {
            List<Elem> A = new ArrayList<>();
            List<Elem> B = new ArrayList<>();
            for (Elem elem : values) {
                if (elem.getTag() == 0) {
                    A.add(new Elem(elem.getTag(), elem.getIndex(), elem.getValue()));
                } else if (elem.getTag() == 1) {
                    B.add(new Elem(elem.getTag(), elem.getIndex(), elem.getValue()));
                }
            }
            for (Elem a : A) {
                for (Elem b : B) {
                    context.write(new Pair(a.getIndex(), b.getIndex()), new DoubleWritable(a.getValue() * b.getValue()));
                }
            }
        }
    }   
    // Second map function that do nothing
    public static class SummationMapper extends Mapper<Object, Text, Pair, DoubleWritable> {
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] line = value.toString().split(",");
            context.write(new Pair(Integer.parseInt(line[0]), Integer.parseInt(line[1])), new DoubleWritable(Double.parseDouble(line[2])));
        }
    }
    // Second reduce function that do the summation
    public static class SummationReducer extends Reducer<Pair, DoubleWritable, Pair, DoubleWritable> {
        @Override
        public void reduce(Pair pair, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double sum = 0.0;
            for (DoubleWritable value : values) {
                sum += value.get();
            }
            context.write(pair, new DoubleWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception {
        // Create first job for performing the first map and reduce
        Job job1 = Job.getInstance();
        job1.setJobName("Matrix Multiplication");
        job1.setJarByClass(Multiply.class);
        Configuration conf1 = job1.getConfiguration();
        conf1.set("mapreduce.output.textoutputformat.separator", ",");
        // Set the mapper and reducer classes
        MultipleInputs.addInputPath(job1, new Path(args[0]), TextInputFormat.class, MatrixMMapper.class);
        MultipleInputs.addInputPath(job1, new Path(args[1]), TextInputFormat.class, MatrixNMapper.class);
        job1.setReducerClass(MatrixReducer.class);
        // Specify the output key and value types for the job's output
        job1.setOutputKeyClass(Pair.class);
        job1.setOutputValueClass(DoubleWritable.class);
        // Specify the output key and value types for the map function
        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(Elem.class);
        job1.setOutputFormatClass(TextOutputFormat.class);
        FileOutputFormat.setOutputPath(job1, new Path(args[2]));
        job1.waitForCompletion(true);

        // Create second job for performing the second map and reduce
        Job job2 = Job.getInstance();
        job2.setJobName("Summation");
        job2.setJarByClass(Multiply.class);
        Configuration conf2 = job2.getConfiguration();
        conf2.set("mapreduce.output.textoutputformat.separator", ",");
        // Set the mapper and reducer classes
        job2.setMapperClass(SummationMapper.class);
        job2.setReducerClass(SummationReducer.class);
        // Set the classes for output keys and values
        job2.setOutputKeyClass(Pair.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setMapOutputKeyClass(Pair.class);
        job2.setMapOutputValueClass(DoubleWritable.class);
        job2.setInputFormatClass(TextInputFormat.class);
        job2.setOutputFormatClass(TextOutputFormat.class);
        FileInputFormat.setInputPaths(job2, new Path(args[2]));
        FileOutputFormat.setOutputPath(job2, new Path(args[3]));
        job2.waitForCompletion(true);
    }
}