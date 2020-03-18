# Casimir: Catalyst, smoothing, and inference
A toolbox of selected optimization algorithms including Casimir-SVRG 
(as well as the special cases of Catalyst-SVRG and SVRG)
for unstructured tasks such as binary classification, and structured prediction tasks 
such as object localization or named entity recognition.
This is code accompanying the paper
"[A Smoother Way to Train Structured Prediction Models](https://krishnap25.github.io/papers/2018_neurips_smoother.pdf)"
 in NeurIPS 2018. 

## Documentation
The documentation for this toolbox can be found [here](https://homes.cs.washington.edu/~pillutla/documentation/casimir/).
Note that to compile the cython files in `casimir/data/named_entity_recognition`, run
```
`./scripts/compile_cython.sh` 
```

## Contributing

Feel free to submit a feature request, or better still, a pull request. 

## Authors

* [Krishna Pillutla](https://homes.cs.washington.edu/~pillutla/)
* [Vincent Roulet](http://faculty.washington.edu/vroulet/)
* [Sham M. Kakade](https://homes.cs.washington.edu/~sham/)
* [Zaid Harchaoui](http://faculty.washington.edu/zaid/)


## License

This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details

## Cite
If you found this package useful, please cite the following work.

```
@incollection{pillutla-etal:casimir:neurips2018,
title = {A {S}moother {W}ay to {T}rain {S}tructured {P}rediction {M}odels},
author = {Pillutla, Krishna and
          Roulet, Vincent and 
          Kakade, Sham M. and
          Harchaoui, Zaid},
booktitle = {Advances in Neural Information Processing Systems 31},
year = {2018},
}
```

## Acknowledgments
This work was supported by NSF Award CCF-1740551, 
the Washington Research Foundation for innovation in Data-intensive Discovery, 
and the program “Learning in Machines and Brains” of CIFAR.

