# Code Token Removal Framework

Code for compressing Java datasets by systematically removing specific types of tokens or iteratively remove tokens in specific order.

## Project Structure

```
codezip
  └──  src/
        └──  main/
              └── java/
                    └── org.example/
                            ├── RemoveType.java       # specified-type token removal
                            ├── TypeDictionary.java   # Token priority dictionary
                            ├── MyVisitor.java        # JavaParser visitor implementation
                            └── Main.java             # iteratively remove tokens to certain ratio
        └── pom.file                                  # maven configuration
```  

