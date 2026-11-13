document.addEventListener('DOMContentLoaded',()=>{
 const toc=document.querySelector('.toc-board'),main=document.querySelector('#post main');if(!toc||!main)return;
 const hs=[...main.querySelectorAll('h1,h2,h3')];if(!hs.length)return;
 const ul=document.createElement('ul');ul.className='toc-list';
 hs.forEach((h,i)=>{if(!h.id)h.id='section-'+(i+1);const li=document.createElement('li');li.className='toc-list-item';const a=document.createElement('a');a.className='node-name--'+h.tagName;a.textContent=h.textContent;a.href='#'+h.id;li.append(a);ul.append(li);});toc.append(ul);const top=document.querySelector('.top-arrow');if(top){top.classList.add('arrow-open');}
});