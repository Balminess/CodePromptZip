package org.example;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RemoveType {
    public static void markFlag(int[] codeFlag, SpanContent spanContent, int flag, String code, boolean[] otherFlag){
        int startWord = spanContent.startWord;
        int endWord = spanContent.endWord;
        for (int i=startWord;i<=endWord-1;i++){
            codeFlag[i] = flag;
            if (otherFlag!=null){
                otherFlag[i] = true;
            }
        }
    }
    public static ArrayList<Integer> getRemovedIndex(String[] codeSplits,boolean[] codeFlag){
        ArrayList<Integer> removeIndex = new ArrayList<>();
        for (int j = codeSplits.length-1;j>=0;j--){
            if (codeFlag[j]){
                removeIndex.add(j);
            }
        }
        return removeIndex;

    }
    public static String removeCode(String code, Map map,String type){
        List<String> categoryList = List.of("function_invocation", "identifiers", "function_structure");
        String[] codeSplits = code.split(" +");
        int codeLength=codeSplits.length;
        boolean[] typeFlag = new boolean[codeLength];
        int[] codeFlag = new int[codeLength];

        if (type.equals("symbols")) {
            String[] simpleStr = new String[]{"=", "+", "-", "*", "/", "%", "!", ">",  "<", "|", "?", ":", "~", "&", "^", "(",
                    "{", ")", "}", "[", ".", "]", ";", "\"", ",","==","++","--","!=",">=","<=","&&","||","<<",">>",">>>","\'"
            };
            List<String> simpleList = Arrays.asList(simpleStr);
            for(int i = 0;i< codeSplits.length;i++){
                if (simpleList.contains(codeSplits[i])){
                    typeFlag[i] = true;
                }
            }
        }

        else if (type.equals("method_signature")) {
            // 找到第一个 { 的位置
            int bracketIndex = -1;
            for(int i = 0; i < codeSplits.length; i++) {
                if (codeSplits[i].equals("{")) {
                    bracketIndex = i;
                    break;
                }
            }
            // 将 { 之前的所有位置标记为 true
            if (bracketIndex != -1) {
                for(int i = 0; i < bracketIndex; i++) {
                    typeFlag[i] = true;
                }
            }
        }
        else {
        ArrayList<SpanContent> typeList = (ArrayList<SpanContent>) map.get(type);

        for (SpanContent spanContent : typeList){
            markFlag(codeFlag,spanContent,1,code,typeFlag);
        }
        }

        ArrayList<Integer> removedIndex = getRemovedIndex(codeSplits, typeFlag);
        for (int index : removedIndex){
            codeSplits[index] = "";
        }
        String new_code = String.join(" ",codeSplits);

        return new_code;
    }
    public static String remove(String code, String type){
        // 特殊处理 symbols 类型
        MyVisitor myVisitor = new MyVisitor(code);
        CompilationUnit cu = JavaParser.parse(myVisitor.code);
        myVisitor.visit(cu, null);
//        System.out.println(myVisitor.map);

        String removedCode = removeCode(code, myVisitor.map,type);
//        System.out.println(removedCode);
        return removedCode;
    }
    public static List<String> readJsonFile(String fileName) throws IOException {
        List<String> lines = new ArrayList<>();

        // Use Files.lines to read file efficiently
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(fileName))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }
        return lines;
    }
    private static double processMethod(JsonObject jsonObject, String methodKey, String type) {
        String method = jsonObject.get(methodKey).getAsString();
        int originalLength = method.split(" +").length;

        method = remove(method, type);
        jsonObject.addProperty(methodKey, method);

        int newLength = method.split(" +").length;
        return (originalLength - newLength) * 1.0 / originalLength;
    }

    public static void main(String[] args) {

        try {
            long startTime = System.currentTimeMillis();

            String stage = "train";
            List<String> stringList = readJsonFile("/home/pengfei/code/slimcode/dataset/assertion/base/atlas-" + stage + "-m.jsonl");
            FileOutputStream fileOutputStream = new FileOutputStream("/home/pengfei/code/slimcode/dataset/assertion/category/"+ "ag_remove_slimcode_test.jsonl");
            OutputStreamWriter outputStreamWriter = new OutputStreamWriter(fileOutputStream);
            BufferedWriter bufferedWriter = new BufferedWriter(outputStreamWriter);
            double allRemovePercent = 0;
            //"function_invocation","identifiers","function_structure","method_signature","symbols"
            String type = "function_invocation";
            int targetIndex=5;
            int count = 0;
            for(int i=0; i < stringList.size();i++){

                if(i == targetIndex) {  // 只处理目标index
                    String lineStr = stringList.get(i);
                    JsonObject jsonObject = JsonParser.parseString(lineStr).getAsJsonObject();
                    String testMethod = jsonObject.get("test_method").getAsString();
                    System.out.println("Index " + i + ":");
                    System.out.println("focal_method:" + testMethod);
                    System.out.println("compressed:" + remove(testMethod, type));
                    break;  // 找到后就退出循环
                }
                String lineStr = stringList.get(i);
                JsonObject jsonObject = JsonParser.parseString(lineStr).getAsJsonObject();

                try {
                    allRemovePercent += processMethod(jsonObject, "focal_method", type);
                    allRemovePercent += processMethod(jsonObject, "test_method", type);
                    count++;
                } catch (ParseProblemException e) {
                    continue;
                }

//            System.out.println("cut:"+code.split(" +").length);
            }
            bufferedWriter.close();
            outputStreamWriter.close();
            System.out.println(count+"/"+stringList.size());
            System.out.println("共有"+count+"条数据");
            double avgRemovePercent = allRemovePercent / count;
            System.out.println("avg remove percent:"+avgRemovePercent);
            long endTime = System.currentTimeMillis();
            System.out.println("totalTime:" + (endTime - startTime));

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}