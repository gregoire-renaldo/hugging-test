from transformers import pipeline

summary = pipeline("summarization")

res = summary("The SOLID principles are a set of design guidelines in object-oriented programming (OOP) that help developers create software that is maintainable, scalable, and robust. The acronym SOLID stands for five principles: Single Responsibility Principle (SRP) Definition: A class should have only one reason to change, meaning it should have only one responsibility or purpose. Explanation: Each class in a system should focus on a single task or responsibility. This reduces the risk of making unintended changes in unrelated parts of the system when modifying the class. Benefits: Improves readability and maintainability. Reduces the risk of bugs when changes are made. Makes classes easier to test. Example: Instead of a User class that handles both user data and database operations, separate the responsibilities into a User class for user-related data and a UserRepository class for database operations. Open Closed Principle (OCP) Definition: A class should be open for extension but closed for modification. Explanation: You should be able to add new functionality to a class without altering its existing code. This can be achieved through abstraction and inheritance or by using composition and interfaces. Benefits: Enhances flexibility and scalability. Reduces the risk of introducing bugs in existing code. Encourages adherence to the don't repeat yourself (DRY) principle. Example: A payment system could have a base Payment class and extend it with CreditCardPayment, PayPalPayment, etc., without modifying the original Payment class. Liskov Substitution Principle (LSP) Definition: Subtypes must be substitutable for their base types without altering the correctness of the program. Explanation: If class B is a subclass of class A, objects of type A should be replaceable with objects of type B without causing errors. Subclasses should only extend behavior and not override it in ways that violate expectations. Benefits: Ensures that derived classes are compatible with base classes. Promotes code reliability and reuse. Example: If Bird is a base class and Penguin is a derived class, and Bird has a method fly(), Penguin should not violate expectations by making fly() invalid. Instead, a different design might be used to accommodate non-flying birds. Interface Segregation Principle (ISP) Definition: A class should not be forced to implement interfaces it does not use. Explanation: Large interfaces should be split into smaller, more specific ones. This ensures that classes only implement methods relevant to them, avoiding unnecessary dependencies. Benefits: Reduces the impact of changes to an interface. Enhances clarity and cohesion. Makes classes easier to maintain and test. Example: Instead of a large Animal interface with methods like run(), fly(), and swim(), create smaller interfaces such as Runnable, Flyable, and Swimmable. Classes like Dog would implement only Runnable, while Fish would implement Swimmable. Dependency Inversion Principle (DIP) Definition: High-level modules should not depend on low-level modules. Both should depend on abstractions. Explanation: Rather than tightly coupling high-level code to low-level details, introduce abstractions (e.g., interfaces) to decouple them. This makes the system more flexible and easier to modify. Benefits: Reduces tight coupling between components. Makes the system more modular and testable. Promotes flexibility and scalability. Example: Instead of a Database class being directly used by a UserService class, define an IDatabase interface. The UserService depends on IDatabase, allowing different database implementations to be swapped in without modifying the UserService."
)

print(res)

# Output: [{'summary_text': ' The SOLID principles are a set of design guidelines in object-oriented programming . They help developers create software that is maintainable, scalable, and robust . The acronym SOLID stands for five principles: Single Responsibility Principle, Open Closed Principle, Dependency Inversion Principle, Liskov Substitution Principle and Interface Segregation Principle .'}]

res2 = summary("The SOLID principles are a set of design guidelines in object-oriented programming (OOP) that help developers create software that is maintainable, scalable, and robust. The acronym SOLID stands for five principles: Single Responsibility Principle (SRP) Definition: A class should have")