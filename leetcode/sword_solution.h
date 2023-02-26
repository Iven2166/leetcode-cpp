//
//  sword_solution.h
//  leetcode
//
//  Created by  on 2023/1/5.
//

#ifndef sword_solution_h
#define sword_solution_h

class sword_solution {
public:
    // 面试题18: 删除链表的节点
    void DeleteNode(ListNode** pListHead, ListNode* pToBeDeleted){
        // 其一为空的
        if(!pListHead || !pToBeDeleted){
            return ;
        }
        // 要删除的节点不是尾节点
        if(pToBeDeleted->next != nullptr){
            ListNode* pNext = pToBeDeleted->next;
            pToBeDeleted->val = pNext->val;
            pToBeDeleted->next = pNext->next;
            
            delete pNext;
            pNext = nullptr;
        }
        // 链表只有一个节点，删除头节点
        else if(*pListHead == pToBeDeleted){
            delete pToBeDeleted;
            pToBeDeleted = nullptr;
            *pListHead = nullptr;
        }
        // 链表有多个节点，删除尾节点（pToBeDeleted->next==nullptr)
        else{
            ListNode* pNode = *pListHead;
            while(pNode->next != pToBeDeleted){
                pNode = pNode->next;
            }
            pNode->next = nullptr;
            delete pToBeDeleted;
            pToBeDeleted = nullptr;
        }
    }
    
    // 面试题17:打印从1到最大的n位数
    void Print1ToMaxNDigits2(int n){
        if(n <= 0)
            return ;
        char* number = new char[n + 1];
        number[n] = '\0';
        
        for(int i = 0; i < 10; i++){
            number[0] = i + '0';
            Print1ToMaxNDigitsRecursively(number, n, 0);
        }
        delete [] number;
    }
    void Print1ToMaxNDigitsRecursively(char* number, int length, int index){
        // index 是定义第几位
        if(index == length - 1){
            Print1ToMaxNDigitsPrint(number);return ;
        }
        
        for(int i = 0; i < 10; i++){
            number[index + 1] = i + '0';
            // 安排下一位
            Print1ToMaxNDigitsRecursively(number, length, index + 1);
        }
    }
    
    
    
    void Print1ToMaxNDigits1(int n){
        if(n <= 0)
            return ;
        char* number = new char[n+1]; // 有个结束符号
        memset(number, '0', n);
        number[n] = '\0';
        while(!Print1ToMaxNDigitsIncreament(number))
            Print1ToMaxNDigitsPrint(number); // 打印这个数
        delete []number;
    }
    
    bool Print1ToMaxNDigitsIncreament(char* number){
        bool isOverflow = false;
        int nTakeOver = 0;
        int nLength = strlen(number);
        for(int i = nLength - 1; i >= 0; i--){
            int nSum = number[i] - '0' + nTakeOver;
            if(i == nLength - 1){
                nSum++;
            }
            if(nSum >= 10){
                if(i == 0){
                    isOverflow = true;
                }
                else{
                    nSum -= 10;
                    nTakeOver = 1;
                    number[i] = '0' + nSum;
                }
            }
            else{
                number[i] = '0' + nSum;
                break;
            }
        }
        return isOverflow;
    }
    
    void Print1ToMaxNDigitsPrint(char* number){
        bool isBeginning0 = true;
        int nLength = strlen(number);
        
        for(int i = 0; i < nLength; ++i){
            if(isBeginning0 && number[i] != '0'){
                isBeginning0 = false;
            }
            if(!isBeginning0){
                printf("%c", number[i]);
            }
        }
        printf("\t");
    }
    
    // 剑指 Offer 16.数值的整数次方
    double Power(double base, int exponent){
        // 非法输入
        if(base < std::numeric_limits<double>::epsilon() && exponent < 0){
            return 0.0;
        }
        unsigned int absExponent = (unsigned int) exponent;
        if(exponent < 0)
            absExponent = (unsigned int) -exponent;
        double res = PowerCore(base, absExponent);
        if(exponent < 0)
            res = 1.0 / res;
        return res;
    }
    double PowerCore(double base, unsigned int absExponent){
        // 普通解法
//        double res = 1.0;
//        for(int i = 0; i < absExponent; i++){
//            res *= base;
//        }
//        return res;
        // 优化解法： a^n = a^(n/2) * a^(n/2); a^n = a^(n/2) * a^(n/2) * a;
        if(absExponent == 0){
            return 1.0;
        }
        if(absExponent == 1){
            return base;
        }
        double mid = PowerCore(base, absExponent >> 1);
        mid = mid * mid;
        if(absExponent & 0x1 == 1)
            mid *= base;
        return mid;        
    }
    
    int hammingWeight(uint32_t n){
        int ans = 0;
        while(n){
            ans += (n & 1);
            n >>= 1;
        }
        
        return ans;
    }
    
    int cuttingRope(int n){
        vector<int> dp(n+1);
        for(int i=2;i<n;i++){
            int curMax = 0;
            for(int j=1; j<i; j++){
                curMax = max(curMax, max(j*(i-j), j*dp[i-j]));
            }
            dp[i] = curMax;
        }
        return dp[n];
    }
};


#endif /* sword_solution_h */
