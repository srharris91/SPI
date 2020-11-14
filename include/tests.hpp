#ifndef TESTS_H
#define TESTS_H
void test_if_true(PetscBool test,std::string name);
void test_if_close(PetscScalar value,PetscScalar golden, std::string name, PetscReal tol=1.E-13);
int tests();
#endif // TESTS_H
