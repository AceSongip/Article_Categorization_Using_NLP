"?:
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1    ^?AA    ^?AaeYR?R??ieYR?R???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff ?@9fffff ?@Afffff ?@Ifffff ?@a??????v?i??
????Unknown?
dHostDataset"Iterator::Model(1fffff??@9fffff??@A????̘?@I????̘?@a???J_?i䃔'?????Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1     hy@9     hy@A     hy@I     hy@a???W?i???????Unknown
?HostResourceGather"'sequential_2/embedding/embedding_lookup(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@a?????,V?iK`o?????Unknown
?HostVariableShape"Cgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape(1fffffnt@9fffffnt@Afffffnt@Ifffffnt@a&?TBS?i^n?2I????Unknown
kHostUnique"Adam/Adam/update/Unique(1     xs@9     xs@A     xs@I     xs@aN?XZR?i?:;v????Unknown
?	HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1????̬j@9????̬j@A????̬j@I????̬j@a?oK?%I?i?? }?????Unknown
g
HostMul"Adam/Adam/update/mul_1(1????̌j@9????̌j@A????̌j@I????̌j@a?????I?i?υ4????Unknown
^HostGatherV2"GatherV2(1fffff?h@9fffff?h@Afffff?h@Ifffff?h@aiu??ǇG?iY&?????Unknown
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1fffff?e@9fffff?e@Afffff?e@Ifffff?e@aBυFDJD?iͮ???????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(133333?d@933333?d@A33333?d@I33333?d@a?L?.?C?i?L??????Unknown
gHostMul"Adam/Adam/update/mul_4(1     pb@9     pb@A     pb@I     pb@av??Q6aA?i?u??.????Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1?????la@9?????la@A?????la@I?????la@a?j???l@?i?r\	J????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1??????^@9??????^@A??????^@I??????^@a??'?=?ipc???????Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(133333?]@933333?]@A33333?]@I33333?]@a???T;<?iXTt????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1fffff?[@9fffff?[@Afffff?[@Ifffff?[@a?q?fy.:?i +?ӹ????Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1fffffU@9fffffU@AfffffU@IfffffU@a.z/??3?ic4????Unknown
fHost_Send"IteratorGetNext/_13(133333sR@933333sR@A33333sR@I33333sR@a?C??:d1?iK?W?`????Unknown
~Host_Send"+sequential_2/embedding/embedding_lookup/_25(133333sN@933333sN@A33333sN@I33333sN@a?Ͱ???,?iX?"?+????Unknown
eHost
LogicalAnd"
LogicalAnd(1     ?L@9     ?L@A     ?L@I     ?L@a??"-?+?i??5j?????Unknown?
eHostMul"Adam/Adam/update/mul(133333sE@933333sE@A33333sE@I33333sE@af??+8$?iZ??? ????Unknown
mHostRealDiv"Adam/Adam/update/truediv(1??????@@9??????@@A??????@@I??????@@a?}DA??iB??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????Y@@9?????Y@@A?????Y@@I?????Y@@a??e???io_y?????Unknown
gHostMul"Adam/Adam/update/mul_2(1???????@9???????@A???????@I???????@aJ??????i???t????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1ffffff=@9ffffff=@Affffff=@Iffffff=@a??~????i?k)?????Unknown
gHostMul"Adam/Adam/update/mul_3(1333333=@9333333=@A333333=@I333333=@azuX??i?.\?????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1?????L9@9?????L9@A?????L9@I?????L9@aJ-?Z9??iv??%~????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?????<@9?????<@A?????7@I?????7@aZ)Q?U??i???X,????Unknown
`HostGatherV2"
GatherV2_1(1??????6@9??????6@A??????6@I??????6@aGnE??M?i*??????Unknown
{ Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1ffffff5@9ffffff5@Affffff5@Iffffff5@a}??@,?iA?&x????Unknown
g!HostAddV2"Adam/Adam/update/add(133333?4@933333?4@A33333?4@I33333?4@a?L?.??i?|^@????Unknown
["HostPow"
Adam/Pow_3(133333?1@933333?1@A33333?1@I33333?1@aR?c>??iΔQ??????Unknown
i#HostWriteSummary"WriteSummary(1333333+@9333333+@A333333+@I333333+@aȍ???	?i`?1I ????Unknown?
g$HostMul"Adam/Adam/update/mul_5(1??????*@9??????*@A??????*@I??????*@a?F?c?	?i????d????Unknown
?%Host_Send"Fgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape/_9(1??????)@9??????)@A??????)@I??????)@a^??q?Q?i?Nq??????Unknown
?&Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1ffffff(@9ffffff(@Affffff(@Iffffff(@a?B	?
 ?i?z??!????Unknown
?'HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????0@9??????0@A??????'@I??????'@an?\??>?ib???z????Unknown
p(Host_Recv"Adam/Cast_6/ReadVariableOp/_6(1??????$@9??????$@A??????$@I??????$@a??Ok?i?Ʉ?????Unknown
p)Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1ffffff#@9ffffff#@Affffff#@Iffffff#@a.???yI?i???????Unknown
l*HostIteratorGetNext"IteratorGetNext(1      #@9      #@A      #@I      #@a?Ľk???i^~NY????Unknown
[+HostSub"
Adam/sub_7(1ffffff!@9ffffff!@Affffff!@Iffffff!@a?e??f ?i????????Unknown
p,Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1?????? @9?????? @A?????? @I?????? @a??G???>i&?#B?????Unknown
x-HostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffK@9ffffffK@A      @I      @a?ؿfeG?>i????????Unknown
f.Host_Send"IteratorGetNext/_11(1      @9      @A      @I      @a??a?$??>ij?7?C????Unknown
?/Host_Recv"Cgradient_tape/sequential_2/embedding/embedding_lookup/Reshape_1/_28(1??????@9??????@A??????@I??????@a7r?C?`?>i)~Z?r????Unknown
?0Host_Recv"Agradient_tape/sequential_2/embedding/embedding_lookup/Reshape/_34(1??????@9??????@A??????@I??????@a?-}]\?>i?xO?????Unknown
?1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a;ՙC??>i/???????Unknown
]2HostCast"Adam/Cast_5(1??????@9??????@A??????@I??????@a?? ?y?>i-???????Unknown
]3HostAddV2"
Adam/add_1(1333333@9333333@A333333@I333333@a??(?6?>i>?)
????Unknown
?4HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a??(?6?>iю?[.????Unknown
[5HostPow"
Adam/Pow_2(1ffffff@9ffffff@Affffff@Iffffff@a??????>ign?K????Unknown
]6HostSqrt"Adam/Sqrt_1(1??????	@9??????	@A??????	@I??????	@a?Ћ5?!?>i??{%c????Unknown
k7Host_Recv"Adam/ReadVariableOp_1/_2(1      @9      @A      @I      @a?3R???>i&???y????Unknown
[8HostMul"
Adam/mul_1(1      @9      @A      @I      @a?3R???>iYH?d?????Unknown
x9HostStridedSlice"Adam/Adam/update/strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a?V?nj?>i3?????Unknown
c:HostRealDiv"Adam/truediv_1(1??????@9??????@A??????@I??????@a????P??>i?B??????Unknown
?;HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(13333333@93333333@A??????@I??????@a????P??>i7Ώ??????Unknown
r<Host_Recv"sequential_2/embedding/Cast/_24(1ffffff@9ffffff@Affffff@Iffffff@a~|?)X?>i????????Unknown
[=HostSub"
Adam/sub_6(1ffffff??9ffffff??Affffff??Iffffff??a?V?nj?>i ?n??????Unknown
[>HostSub"
Adam/sub_4(1????????9????????A????????I????????a????P??>i?m?????Unknown
[?HostSub"
Adam/sub_5(1      ??9      ??A      ??I      ??a????*?>i?r???????Unknown
a@HostIdentity"Identity(1????????9????????A????????I????????a?Ћ5?!?>i      ???Unknown?*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1fffff ?@9fffff ?@Afffff ?@Ifffff ?@a??6?????i??6??????Unknown?
dHostDataset"Iterator::Model(1fffff??@9fffff??@A????̘?@I????̘?@a??{&L??i?????|???Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(1     hy@9     hy@A     hy@I     hy@a?n?IJ???ik?B}????Unknown
?HostResourceGather"'sequential_2/embedding/embedding_lookup(1fffff?w@9fffff?w@Afffff?w@Ifffff?w@a{??,?`??i?:w?Q???Unknown
?HostVariableShape"Cgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape(1fffffnt@9fffffnt@Afffffnt@Ifffffnt@a???#s??in???G???Unknown
kHostUnique"Adam/Adam/update/Unique(1     xs@9     xs@A     xs@I     xs@a???O	??i??ɓ?????Unknown
?HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1????̬j@9????̬j@A????̬j@I????̬j@a"l?m???i???e?"???Unknown
gHostMul"Adam/Adam/update/mul_1(1????̌j@9????̌j@A????̌j@I????̌j@a????%|??iy?Ⱦ?J???Unknown
^	HostGatherV2"GatherV2(1fffff?h@9fffff?h@Afffff?h@Ifffff?h@a?Ww?2a??i?a???`???Unknown
{
HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1fffff?e@9fffff?e@Afffff?e@Ifffff?e@a??Qk???i??J&`P???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(133333?d@933333?d@A33333?d@I33333?d@aE???Ҝ?i>q???6???Unknown
gHostMul"Adam/Adam/update/mul_4(1     pb@9     pb@A     pb@I     pb@a]m???i'??[???Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1?????la@9?????la@A?????la@I?????la@a??)?C??i7??w????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1??????^@9??????^@A??????^@I??????^@a?????z??iƚ??Kr???Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(133333?]@933333?]@A33333?]@I33333?]@aI??%ڔ?i??Q????Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1fffff?[@9fffff?[@Afffff?[@Ifffff?[@a??*|V??i|??г???Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1fffffU@9fffffU@AfffffU@IfffffU@a?^?/?F??i?a??(???Unknown
fHost_Send"IteratorGetNext/_13(133333sR@933333sR@A33333sR@I33333sR@a2???ᰉ?iyӇN?????Unknown
~Host_Send"+sequential_2/embedding/embedding_lookup/_25(133333sN@933333sN@A33333sN@I33333sN@aK?ϯC3??izG]|????Unknown
eHost
LogicalAnd"
LogicalAnd(1     ?L@9     ?L@A     ?L@I     ?L@aw???C??i0?
l?4???Unknown?
eHostMul"Adam/Adam/update/mul(133333sE@933333sE@A33333sE@I33333sE@aK?./K?}?i?#iJp???Unknown
mHostRealDiv"Adam/Adam/update/truediv(1??????@@9??????@@A??????@@I??????@@aYSrq?vw?ihL7????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1?????Y@@9?????Y@@A?????Y@@I?????Y@@aU??5K?v?iķ??????Unknown
gHostMul"Adam/Adam/update/mul_2(1???????@9???????@A???????@I???????@a?g??#v?i??Qk????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1ffffff=@9ffffff=@Affffff=@Iffffff=@a>?
xt?ib?g??!???Unknown
gHostMul"Adam/Adam/update/mul_3(1333333=@9333333=@A333333=@I333333=@aF%VexTt?i?;2??J???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1?????L9@9?????L9@A?????L9@I?????L9@a?A?IZ?q?i0??L?m???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?????<@9?????<@A?????7@I?????7@a?0-<p?i???????Unknown
`HostGatherV2"
GatherV2_1(1??????6@9??????6@A??????6@I??????6@a???<xo?ix?>~????Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1ffffff5@9ffffff5@Affffff5@Iffffff5@a??l\x?m?i4?yJ????Unknown
gHostAddV2"Adam/Adam/update/add(133333?4@933333?4@A33333?4@I33333?4@aE????l?iB?pj????Unknown
[ HostPow"
Adam/Pow_3(133333?1@933333?1@A33333?1@I33333?1@a,&Up??h?ih*??? ???Unknown
i!HostWriteSummary"WriteSummary(1333333+@9333333+@A333333+@I333333+@a=--???b?i?W??????Unknown?
g"HostMul"Adam/Adam/update/mul_5(1??????*@9??????*@A??????*@I??????*@a?????b?ix?? 8&???Unknown
?#Host_Send"Fgradient_tape/sequential_2/embedding/embedding_lookup/VariableShape/_9(1??????)@9??????)@A??????)@I??????)@a??gx?a?i??4y.8???Unknown
?$Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1ffffff(@9ffffff(@Affffff(@Iffffff(@a??????`?i?|j+I???Unknown
?%HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1??????0@9??????0@A??????'@I??????'@a?n?JZn`?i9``ęY???Unknown
p&Host_Recv"Adam/Cast_6/ReadVariableOp/_6(1??????$@9??????$@A??????$@I??????$@ax?K0K?\?i4??i?g???Unknown
p'Host_Recv"Adam/Cast_4/ReadVariableOp/_4(1ffffff#@9ffffff#@Affffff#@Iffffff#@a??n?[?i???-su???Unknown
l(HostIteratorGetNext"IteratorGetNext(1      #@9      #@A      #@I      #@a?h
??tZ?iΘ??????Unknown
[)HostSub"
Adam/sub_7(1ffffff!@9ffffff!@Affffff!@Iffffff!@a????:X?i<}[?ʎ???Unknown
p*Host_Recv"Adam/Cast_7/ReadVariableOp/_8(1?????? @9?????? @A?????? @I?????? @a?F???dW?i_ժK}????Unknown
x+HostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffffK@9ffffffK@A      @I      @a|?f??T?i??(??????Unknown
f,Host_Send"IteratorGetNext/_11(1      @9      @A      @I      @al?R?i?/??????Unknown
?-Host_Recv"Cgradient_tape/sequential_2/embedding/embedding_lookup/Reshape_1/_28(1??????@9??????@A??????@I??????@a??+<DQ?i?E ?????Unknown
?.Host_Recv"Agradient_tape/sequential_2/embedding/embedding_lookup/Reshape/_34(1??????@9??????@A??????@I??????@a??t??N?i,?.?"????Unknown
?/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a?`3Oi?K?i??K????Unknown
]0HostCast"Adam/Cast_5(1??????@9??????@A??????@I??????@ap?"??JK?i?1 ?????Unknown
]1HostAddV2"
Adam/add_1(1333333@9333333@A333333@I333333@a9?#<?J?ij?9?????Unknown
?2HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a9?#<?J?i?BJ????Unknown
[3HostPow"
Adam/Pow_2(1ffffff@9ffffff@Affffff@Iffffff@a?nFZ*E?i?(Դ?????Unknown
]4HostSqrt"Adam/Sqrt_1(1??????	@9??????	@A??????	@I??????	@a?f???A?iګ?i	????Unknown
k5Host_Recv"Adam/ReadVariableOp_1/_2(1      @9      @A      @I      @ad?땥?@?i?&??6????Unknown
[6HostMul"
Adam/mul_1(1      @9      @A      @I      @ad?땥?@?i??O<d????Unknown
x7HostStridedSlice"Adam/Adam/update/strided_slice(1ffffff@9ffffff@Affffff@Iffffff@a쳕??0??i`jZJ????Unknown
c8HostRealDiv"Adam/truediv_1(1??????@9??????@A??????@I??????@a'T{??<?i?~9-?????Unknown
?9HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(13333333@93333333@A??????@I??????@a'T{??<?ij? ?????Unknown
r:Host_Recv"sequential_2/embedding/Cast/_24(1ffffff@9ffffff@Affffff@Iffffff@a?????9?i?????????Unknown
[;HostSub"
Adam/sub_6(1ffffff??9ffffff??Affffff??Iffffff??a쳕??0/?i? ???????Unknown
[<HostSub"
Adam/sub_4(1????????9????????A????????I????????a'T{??,?iA?\Z~????Unknown
[=HostSub"
Adam/sub_5(1      ??9      ??A      ??I      ??a???r?G&?i9????????Unknown
a>HostIdentity"Identity(1????????9????????A????????I????????a?f???!?i?????????Unknown?