
//
//  leetcode-solution.cpp
//  cpp-demo
//
//  Created by tmp on 2022/11/30.
//

//#include "leetcode-solution.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>
#include "Solution.hpp"
#include "sword_solution.h"

using namespace std;


int main(){
    Solution solution;
    
    cout << "----- no.20 -----";
    cout << endl;
    string s20 = "()[]{";
    bool res20 = solution.isValid(s20);
    cout << res20 << endl;
    
    cout << "----- no.3 -----";
    cout << endl;
    string s3 = "abcabcbb";
    int res3 = solution.lengthOfLongestSubstring(s3);
    cout << res3 << endl;
    
    cout << "----- no.2 -----";
    cout << endl;
    ListNode* l1 = new ListNode(-1);
    ListNode* node = l1;
    for(auto& it: {2,4,3}){
        node->next = new ListNode(it);
        node = node->next;
    }
    ListNode* l2 = new ListNode(-1);
    node = l2;
    for(auto& it: {5,6,4}){
        node->next = new ListNode(it);
        node = node->next;
    }
    ListNode* res2 = solution.addTwoNumbers(l1->next,l2->next);
    while(res2){
        cout << res2->val;
        res2 = res2->next;
    }
    
    cout << "----- no.4 -----";
    cout << endl;
    vector<int> nums1 = {1,2,3,5,6,10};
    vector<int> nums2 = {3,4};
    double res4 = solution.findMedianSortedArrays(nums1, nums2);
    cout << res4 << endl;
    
    
    cout << "----- no.1 -----";
    cout << endl;
    vector <int> nums = {2,7,11,15};
    int target = 26;
    vector<int> res1 = solution.twoSum(nums, target);
    cout << res1[0] << ','<< res1[1] << endl;
    
    
    cout << "----- no.5 -----";
    cout << endl;
    string s = "babad";
    string res5 = solution.longestPalindrome(s);
    for(int i=0; i<res5.size(); i++){
        std::cout << res5[i];
    }
    cout << endl;
    
    cout << "----- no.15 -----";
    cout << endl;
    vector<int> s15 = {-1,0,1,2,-1,-4}; // -4,-1,-1,0,1,2
    //    vector<int> s15 = {0,0,0,0};//
    vector<vector<int>> res15 = solution.threeSum(s15);
    for(auto& iter1: res15){
        for(auto& iter2: iter1){
            cout << iter2 << ',';
        }
        cout << endl;
    }
    cout << endl;
    //    cout << res15 << endl;
    
    cout << "----- no.42 -----";
    cout << endl;
    vector<int> s42 = {0,1,0,2,1,0,1,3,2,1,2,1};//{4,2,0,3,2,5};//{0,1,0,2,1,0,1,3,2,1,2,1};
    int res42 = solution.trap(s42);
    cout << res42 << endl;
    
    cout << "----- no.200 -----";
    vector<vector<char>> grid = {{'1','1','1','1','0'},{'1','1','0','1','0'},{'1','1','0','0','0'},{'0','0','0','0','0'}};
    int res200 = solution.numIslands(grid);
    cout << res200 << endl;
    
    
    cout << "----- no.22 -----" << endl;
    int input22 = 7;
    vector<string> res22 = solution.generateParenthesis(input22);
    for(auto& iter: res22)
        cout << iter <<endl;
    
    cout << "----- no.11 -----" << endl;
    vector<int> input11 = {1,8,6,2,5,4,8,3,7};
    int res11 = solution.maxArea(input11);
    cout << res11 << endl;
    
    cout << "----- no.72 -----" << endl;
    string word1="horse";
    string word2="ros";
    int res72 = solution.minDistance(word1, word2);
    cout << res72 << endl;
    
    cout << "----- no.21 -----" << endl;
    ListNode* input21_p1 = new ListNode(1, new ListNode(2, new ListNode(4)));
    ListNode* input21_p2 = new ListNode(1, new ListNode(3, new ListNode(4)));
    ListNode* res21 = solution.mergeTwoLists(input21_p1, input21_p2);
    ListNode* node21 = res21;
    while(node21){
        cout << node21->val << ',';
        node21 = node21->next;
    }
    
    cout << "----- no.53 -----" << endl;
    vector<int> input53 = {5,4,-1,7,8};
    int res53 = solution.maxSubArray(input53);
    cout << res53 << endl;
    
    cout << "----- no.416 -----" << endl;
    vector<int> input416 = {99,1};
    int res416 = solution.canPartition(input416);
    cout << res416 << endl;
    
    cout << "----- no.698 -----" << endl;
    vector<int> input698_1 = {1,2,3,4};//{4, 3, 2, 3, 5, 2, 1};
    int input698_2 = 4;
    bool res698 = solution.canPartitionKSubsets(input698_1, input698_2);
    cout << res698 << endl;
    
    cout << "----- no.394 -----" << endl;
    string input394 = "3[a]2[bc]";
    string res394 = solution.decodeString(input394);
    cout << res394 << endl;
    
    cout << "----- no.31 -----" << endl;
    vector<int> input31 = {3,1,4,2};
    solution.nextPermutation(input31);
    for(auto i:input31)
        cout << i;
    cout << endl;
    
    int tmp[] = {1,2,3,4,5};
    //    vector<int> tmp = {1,2,3,4,5};
    cout << solution.lowerBound(tmp, 0, 5, 3)<<endl;
    cout << solution.higherBound(tmp, 0, 5, 3)<<endl;
    
    cout << "----- no.归并排序 -----" << endl;
    vector<int> inputGuibing = {3,1,2,4,5,8,7,6};
    solution.guibingMergeSort(inputGuibing, 0, inputGuibing.size()-1);
    for(auto i:inputGuibing ){
        cout << i;
    }
    cout << endl;
    
    cout << "----- no.301.删除无效的括号 -----" << endl;
    string input301 = "()())()";
    vector<string> ans301 = solution.removeInvalidParentheses1(input301);
    for(auto i: ans301)
        cout << i << endl;
    cout << endl;
    
    cout << "----- no.112.路径总和 -----" << endl;
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(3);
    cout << solution.hasPathSum(root, 3) << endl;
    cout << solution.hasPathSumBFS(root, 3) << endl;
    cout << endl;
    
    cout << "----- 剑指 Offer 16.数值的整数次方 -----" << endl;
    sword_solution sword_solu;
    double sword_res13 = sword_solu.Power(4.0, 13);
    cout << sword_res13 << endl;
    
    cout << "----- 剑指 Offer 17.打印大数 -----" << endl;
//    sword_solu.Print1ToMaxNDigits1(2);
    sword_solu.Print1ToMaxNDigits2(2);
    cout << endl;
    
    
    cout << "单调栈" << endl;
//    vector<int> input_dandiao = {2,1,2,4,3};
    int input_dandiao[] = {2,1,2,4,3};
//    for(auto i: solution.nextLargerElement(input_dandiao)){
//        cout << i << ",";
//    }
    int* output_dandiao;
    output_dandiao = solution.nextLargerElement(input_dandiao, sizeof(input_dandiao)/sizeof(int));
    for(int i=0; i<5; i++){
        cout << output_dandiao[i] << ",";
    }
    cout << endl;
    
    cout << "739. 每日温度" << endl;
    vector<int> input739 = {73,74,75,71,69,72,76,73};
    vector<int> res739 = solution.dailyTemperatures(input739);
    for(int i=0; i<res739.size(); i++){
        cout << res739[i] << ',';
    }
    cout << endl;
    
    cout << "503. 下一个更大元素 II" << endl;
    vector<int> input503 = {1,2,1}; // 应该返回 [2,-1,2]
    for(auto i: solution.nextGreaterElement2(input503))
        cout << i << ',';
    cout << endl;
    
    cout << "698. 划分为k个相等的子集" << endl;
    vector<int> input698 = {4, 3, 2, 3, 5, 2, 1}; // 返回true， 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和
    int k698 = 4;
    bool output698 = solution.canPartitionKSubsets698(input698, k698);
    cout << output698 << endl;
    
    bool output698_frombuckets = solution.canPartitionKSubsets698_fromBuckets(input698, k698);
    cout << output698_frombuckets << endl;
    
    cout << "86. 分隔链表" << endl;
    ListNode* input86 = new ListNode(-1);
    ListNode* input86_dummy = input86;
    for(auto& it: {1,6,3,2,5,2}){
        input86_dummy->next = new ListNode(it);
        input86_dummy = input86_dummy->next;
    }
    ListNode* output86 = solution.partition86(input86->next, 3);
    ListNode* output86_dummy = output86;
    while(output86_dummy!=nullptr){
        cout << output86_dummy->val << ',';
        output86_dummy = output86_dummy->next;
    }
    cout << endl;
    
    cout << "实现最小堆、最大堆" << endl;
    /*
     参考
     中文：https://blog.csdn.net/geter_CS/article/details/102580332
     https://cplusplus.com/reference/queue/priority_queue/
     参数：
     第一个参数T：元素（element）类型
     第二个参数Container：必须是容器类型Container，用来存储元素（element），其类型必须是第一个参数
     第三个参数Compare：比较形式，默认是less
     
     方法：
     q.size();//返回q里元素个数
     q.empty();//返回q是否为空，空则返回1，否则返回0
     q.push(k);//在q的末尾插入k
     q.pop();//删掉q的第一个元素
     q.top();//返回q的第一个元素
     */
    priority_queue<int, vector<int>, less<int>> big_heap;
    for(auto& i: {2,3,6,1,0}){
        big_heap.push(i);
//        cout << i;
    }
    cout << "size" << big_heap.size() << endl;
    while(!big_heap.empty()){
            cout<<big_heap.top()<<endl;
            big_heap.pop();
    }
    priority_queue<int, vector<int>, greater<int>> small_heap;
    for(auto& i: {2,3,6,1,0}){
        small_heap.push(i);
//        cout << i;
    }
    cout << "size" << small_heap.size() << endl;
    while(!small_heap.empty()){
            cout<<small_heap.top()<<endl;
        small_heap.pop();
    }
    cout << "自定义优先队列" << endl;
    priority_queue<int, vector<int>, myPriorityQueueCompareOnInts> mypq_type;
    for(auto& i: {2,3,6,1,0}){
        mypq_type.push(i);
    }
    while(!mypq_type.empty()){
        cout<<mypq_type.top()<<endl;
        mypq_type.pop();
    }
    return 0;
}

