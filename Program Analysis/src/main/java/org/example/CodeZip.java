package org.example;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import java.io.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.HashMap;

public class CodeZip {


    public static void markFlag(int[] codeFlag,SpanContent spanContent,int flag){
        int startWord = spanContent.startWord;
        int endWord = spanContent.endWord;
        for (int i=startWord;i<=endWord-1;i++){
            codeFlag[i] = flag;

        }
    }
    public static ArrayList<Integer> getRemovedIndex(String[] codeSplits, int[] codeFlag, int targetLength, int[] freqFlag) {
        ArrayList<Integer> removeIndex = new ArrayList<>();
        int removeTargetLength = codeSplits.length - targetLength;

        for (int i = 0; i < 7; i++) {
            ArrayList<Integer> candidates = new ArrayList<>();

            // Collect the index of the current codeFlag value i
            for (int j = codeSplits.length - 1; j >= 0; j--) {
                if (codeFlag[j] == i) {
                    candidates.add(j);
                }
            }
            // Sort the index by the value of freqFlag (descending)
            candidates.sort((a, b) -> Integer.compare(freqFlag[b], freqFlag[a]));

            for (int index : candidates) {
                if (removeIndex.size() >= removeTargetLength) {
                    return removeIndex;
                }
                removeIndex.add(index);
            }
        }

        return removeIndex;
    }


    public static String removeCode(String code, Map map, int targetLength, TypeDictionary typeValueDict){

        ArrayList<SpanContent> identifierList = (ArrayList<SpanContent>) map.get("identifiers");
        ArrayList<SpanContent> invocationList = (ArrayList<SpanContent>) map.get("function_invocation");
        ArrayList<SpanContent> structureList = (ArrayList<SpanContent>) map.get("function_structure");

        String[] codeSplits = code.split(" +");

        int[] codeFlag = new int[codeSplits.length];

        // 使用 Map 统计每个单词的出现次数
        Map<String, Integer> wordCountMap = new HashMap<>();
        for (String word : codeSplits) {
            wordCountMap.put(word, wordCountMap.getOrDefault(word, 0) + 1);
        }

        // 创建数组并填充对应单词的出现次数
        int[] freqFlag = new int[codeSplits.length];
        for (int i = 0; i < codeSplits.length; i++) {
            freqFlag[i] = wordCountMap.get(codeSplits[i]);
        }

        // signature > identifier > structure > invocation > simple symbols

        for (SpanContent spanContent : identifierList){
            markFlag(codeFlag,spanContent,typeValueDict.getTypeValue("identifiers"));
        }

        for (SpanContent spanContent : structureList){
            markFlag(codeFlag,spanContent,typeValueDict.getTypeValue("function_structure"));
        }
        for (SpanContent spanContent : invocationList){
            markFlag(codeFlag,spanContent,typeValueDict.getTypeValue("function_invocation"));
        }
        // symbols
        String[] simpleStr = new String[]{"=", "+", "-", "*", "/", "%", "!", ">",  "<", "|", "?", ":", "~", "&", "^", "(",
                "{", ")", "}", "[", ".", "]", ";", "\"", ",","==","++","--","!=",">=","<=","&&","||","<<",">>",">>>","\'"
        };
        List<String> simpleList = Arrays.asList(simpleStr);
        for(int i = 0;i< codeSplits.length;i++){
            if (simpleList.contains(codeSplits[i])){
                codeFlag[i] = typeValueDict.getTypeValue("symbols");
            }
        }
        // signture
        int bracketIndex = -1;
        for(int i = 0; i < codeSplits.length; i++) {
            if (codeSplits[i].equals("{")) {
                bracketIndex = i;
                break;
            }
        }

        if (bracketIndex != -1) {
            for(int i = 0; i < bracketIndex; i++) {
                codeFlag[i] = typeValueDict.getTypeValue("method_signature");
            }
        }

        //other
        for (int i = 0; i<codeSplits.length;i++){
            if (codeFlag[i] == 0){
                int start = i;
                while (start < codeFlag.length && codeFlag[start] == 0){
                    start ++;
                }
                int end = start;
                for (int k=i;k<end;k++){
                    codeFlag[k] = 6;
                }
            }
        }

        String removedCode = "";
        ArrayList<Integer> removedIndex = getRemovedIndex(codeSplits, codeFlag,targetLength,freqFlag);
        for (int index : removedIndex){
            removedCode += codeSplits[index] + " ";
            codeSplits[index] = "";
        }
//        System.out.println(removedCode);

        String new_code = String.join(" ",codeSplits);


        return new_code;
    }

    public static String remove(String code,int targetLength,TypeDictionary typeValueDict){

        if (code.split(" +").length <= targetLength){
            return code;
        }

        MyVisitor myVisitor = new MyVisitor(code);
        CompilationUnit cu = JavaParser.parse(myVisitor.code);
        myVisitor.visit(cu, null);

        String removedCode = removeCode(code, myVisitor.map,targetLength, typeValueDict);
//        System.out.println(removedCode);
        return removedCode;
    }

    public static List<String> readJsonlFile(String fileName) throws IOException {
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

    public static void main(String[] args) {
        try {
            List<String> types = List.of("method_signature", "function_invocation", "identifiers", "function_structure", "symbols");
            TypeDictionary typeValueDict = new TypeDictionary(types);

            long startTime = System.currentTimeMillis();

            String inputFilePath = "/home/pengfei/code/slimcode/dataset/assertion/base/atlas-train-m.jsonl";
            String outputFilePath = "/home/pengfei/code/slimcode/dataset/assertion/category/ag_remove_slimcode_0.7.jsonl";

            float ratio = 0.5f;

            // Read input JSON lines
            List<String> stringList = readJsonlFile(inputFilePath);

            // Setup writers
            try (BufferedWriter bufferedWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFilePath)))) {
                double totalRemovePercent = 0;
                int validCount = 0;
//i < stringList.size().replaceAll(",", ".")
                for (int i = 0; i < stringList.size(); i++) {
                    String lineStr = stringList.get(i);
                    try {
                        JsonObject jsonObject = JsonParser.parseString(lineStr).getAsJsonObject();
                        // Process "focal_method"
                        String focalMethod = jsonObject.get("focal_method").getAsString();
                        int originalFocalLength = focalMethod.split(" +").length;
                        int focalTargetLength = Math.round(originalFocalLength * (1 - ratio));
                        focalMethod = remove(focalMethod, focalTargetLength,typeValueDict);
                        jsonObject.addProperty("focal_method", focalMethod);

                        // Calculate removal percentage for "focal_method"
                        int reducedFocalLength = focalMethod.split(" +").length;
                        double focalRemovePercent = (originalFocalLength - reducedFocalLength) * 1.0 / originalFocalLength;
                        totalRemovePercent += focalRemovePercent;

//                         Process "test_method"
                        String testMethod = jsonObject.get("test_method").getAsString();
//                        System.out.println("TEST####\n"+testMethod);
                        int originalTestLength = testMethod.split(" +").length;
                        int testTargetLength = Math.round(originalTestLength * (1 - ratio));
                        testMethod = remove(testMethod, testTargetLength,typeValueDict);
                        jsonObject.addProperty("test_method", testMethod);

                        // Calculate removal percentage for "test_method"
                        int reducedTestLength = testMethod.split(" +").length;
                        double testRemovePercent = (originalTestLength - reducedTestLength) * 1.0 / originalTestLength;
                        totalRemovePercent += testRemovePercent;
//
//                        // Write processed JSON object to output
//                        bufferedWriter.write(jsonObject.toString() + "\n");
                        validCount++;
                    } catch (Exception e) {
                        System.err.println("Error processing line " + i + ": " + e.getMessage());
                    }

                }
                // Print summary
                System.out.println(validCount + "/" + stringList.size() + " lines processed successfully.");
                System.out.println("Total processed entries: " + validCount);
                if (validCount > 0) {
                    double avgRemovePercent = totalRemovePercent / (validCount * 2); // Multiply by 2 for two fields
                    System.out.println("Average removal percentage: " + avgRemovePercent);
                }

                long endTime = System.currentTimeMillis();
                System.out.println("Total time: " + (endTime - startTime) + "ms");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}