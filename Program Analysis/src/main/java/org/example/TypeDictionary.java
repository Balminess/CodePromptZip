package org.example;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TypeDictionary {
    private Map<String, Integer> typeMap;

    // 构造函数，初始化字典
    public TypeDictionary(List<String> types) {
        typeMap = new HashMap<>();
        int index = 1; // 从1开始赋值
        for (String type : types) {
            typeMap.put(type, index);
            index++;
        }
    }

    // 获取指定类型的值
    public int getTypeValue(String type) {
        return typeMap.getOrDefault(type, -1); // 如果不存在返回-1
    }

    public static void main(String[] args) {
        // 初始化类型列表
        List<String> types = List.of("signature", "identifier", "structure", "invocation", "symbols");

        // 创建字典
        TypeDictionary typeDict = new TypeDictionary(types);

        // 示例：获取某个类型的值
        System.out.println("Value of 'signature': " + typeDict.getTypeValue("signature")); // 输出: 1
        System.out.println("Value of 'symbols': " + typeDict.getTypeValue("symbols"));     // 输出: 5
        System.out.println("Value of 'unknown': " + typeDict.getTypeValue("unknown"));     // 输出: -1
    }
}