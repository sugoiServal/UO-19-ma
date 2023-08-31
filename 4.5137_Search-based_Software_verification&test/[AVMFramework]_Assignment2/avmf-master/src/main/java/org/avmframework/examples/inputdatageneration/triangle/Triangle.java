package org.avmframework.examples.inputdatageneration.triangle;

public class Triangle {

  public enum TriangleType {
    NOT_A_TRIANGLE,
    SCALENE,
    EQUILATERAL,
    ISOSCELES;
  }

  public static TriangleType classify(int num1, int num2, int num3) {
    TriangleType type;
    //`1
    if (num1 > num2) {
      int temp = num1;
      num1 = num2;
      num2 = temp;
    }
    //2
    if (num1 > num3) {
      int temp = num1;
      num1 = num3;
      num3 = temp;
    }
    //3
    if (num2 > num3) {
      int temp = num2;
      num2 = num3;
      num3 = temp;
    }
    //4T-END
    if (num1 + num2 <= num3) 
    {
      type = TriangleType.NOT_A_TRIANGLE;
    } //4F
    else 
    {
      type = TriangleType.SCALENE;
      //5T
      if (num1 == num2) 
      {
    	//6T-END
        if (num2 == num3) 
        {
          type = TriangleType.EQUILATERAL;
        } //6F-END
      } //5F 
      else 
      {
    	//7T
        if (num1 == num2) 
        {
          type = TriangleType.ISOSCELES;
        } 
        else if (num2 == num3) 
        {
          type = TriangleType.ISOSCELES;
        }
      }
    }
    return type;
  }
}
