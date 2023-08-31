package org.avmframework.examples.inputdatageneration.triangle;
import org.avmframework.examples.inputdatageneration.triangle.Triangle.TriangleType;

public class TriangleExample {
    public static void main (String[] args) {
    	TriangleType type = Triangle.classify(489, 884, 489); 
        System.out.println(type);
    }
}
